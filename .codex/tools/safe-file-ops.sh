#!/bin/sh
set -eu

usage() {
  cat <<'EOF'
Usage:
  sh safe-file-ops.sh --config <file> <command> [args]

Commands:
  write <path>        Read stdin and atomically write a file
  touch <path>        Create an empty file
  mkdir <path>        Create a directory
  move <src> <dst>    Move or rename a file or directory
  delete <path>       Soft-delete by moving into the configured delete_dir
  help                Show this message

Config format:
  allowed_root=/absolute/path
  allowed_root=/another/absolute/path
  delete_dir=/absolute/path/to/delete
  audit_log=/absolute/path/to/file-ops.log

Notes:
  - Paths in the config must be absolute.
  - delete never removes permanently; it moves the target into delete_dir.
  - write, touch, mkdir, move, and delete are blocked outside allowed_root entries.
EOF
}

fail() {
  echo "$*" >&2
  exit 1
}

require_args() {
  expected="$1"
  shift
  [ "$#" -eq "$expected" ] || fail "Wrong number of arguments for command"
}

resolve_path() {
  input_path="$1"

  case "$input_path" in
    /*)
      raw_path="$input_path"
      ;;
    *)
      current_dir="$(CDPATH= cd . 2>/dev/null && pwd -P)" || fail "Cannot resolve current directory"
      raw_path="$current_dir/$input_path"
      ;;
  esac

  suffix=""
  probe="$raw_path"

  while [ ! -e "$probe" ] && [ "$probe" != "/" ]; do
    probe_base="$(basename "$probe")"
    suffix="/$probe_base$suffix"
    probe="$(dirname "$probe")"
  done

  if [ -d "$probe" ]; then
    base_dir="$probe"
  else
    probe_base="$(basename "$probe")"
    suffix="/$probe_base$suffix"
    base_dir="$(dirname "$probe")"
  fi

  base_abs="$(CDPATH= cd "$base_dir" 2>/dev/null && pwd -P)" || fail "Cannot resolve path: $input_path"

  if [ "$base_abs" = "/" ]; then
    if [ -n "$suffix" ]; then
      printf '/%s\n' "${suffix#/}"
    else
      printf '/\n'
    fi
  else
    printf '%s%s\n' "$base_abs" "$suffix"
  fi
}

find_allowed_root() {
  target_path="$1"
  MATCHED_ROOT=""

  while IFS= read -r root_path || [ -n "$root_path" ]; do
    [ -n "$root_path" ] || continue
    case "$target_path" in
      "$root_path"|"$root_path"/*)
        MATCHED_ROOT="$root_path"
        return 0
        ;;
    esac
  done < "$ROOTS_FILE"

  return 1
}

assert_allowed_path() {
  target_path="$1"
  find_allowed_root "$target_path" || fail "Path is outside allowed roots: $target_path"
}

assert_not_delete_tree() {
  target_path="$1"
  case "$target_path" in
    "$DELETE_DIR"|"$DELETE_DIR"/*)
      fail "Direct operations inside delete_dir are blocked: $target_path"
      ;;
  esac
}

ensure_parent_dir() {
  target_path="$1"
  parent_dir="$(dirname "$target_path")"
  [ -d "$parent_dir" ] || fail "Parent directory does not exist: $parent_dir"
}

sanitize_path_id() {
  printf '%s' "$1" | sed 's#^/##; s#[^A-Za-z0-9._-]#_#g'
}

now_stamp() {
  date '+%Y%m%d-%H%M%S'
}

log_action() {
  action="$1"
  src_path="$2"
  dst_path="$3"

  [ -n "$AUDIT_LOG" ] || return 0

  audit_parent="$(dirname "$AUDIT_LOG")"
  mkdir -p "$audit_parent"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$action" "$src_path" "$dst_path" >> "$AUDIT_LOG"
}

CONFIG_FILE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      [ "$#" -ge 2 ] || fail "Missing value for --config"
      CONFIG_FILE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

[ -n "$CONFIG_FILE" ] || fail "Missing --config"
[ -f "$CONFIG_FILE" ] || fail "Config file not found: $CONFIG_FILE"
[ "$#" -ge 1 ] || fail "Missing command"

command_name="$1"
shift

ROOTS_FILE="$(mktemp)"
cleanup() {
  rm -f "$ROOTS_FILE"
}
trap cleanup EXIT HUP INT TERM

DELETE_DIR=""
AUDIT_LOG=""

while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  case "$raw_line" in
    ''|\#*)
      continue
      ;;
  esac

  key="${raw_line%%=*}"
  value="${raw_line#*=}"

  [ "$key" != "$value" ] || fail "Invalid config line: $raw_line"
  [ -n "$value" ] || fail "Empty config value for key: $key"

  case "$key" in
    allowed_root)
      case "$value" in
        /*) : ;;
        *) fail "allowed_root must be absolute: $value" ;;
      esac
      resolved_root="$(resolve_path "$value")"
      [ -d "$resolved_root" ] || fail "allowed_root must exist and be a directory: $resolved_root"
      printf '%s\n' "$resolved_root" >> "$ROOTS_FILE"
      ;;
    delete_dir)
      case "$value" in
        /*) : ;;
        *) fail "delete_dir must be absolute: $value" ;;
      esac
      DELETE_DIR="$(resolve_path "$value")"
      ;;
    audit_log)
      case "$value" in
        /*) : ;;
        *) fail "audit_log must be absolute: $value" ;;
      esac
      AUDIT_LOG="$(resolve_path "$value")"
      ;;
    *)
      fail "Unknown config key: $key"
      ;;
  esac
done < "$CONFIG_FILE"

[ -s "$ROOTS_FILE" ] || fail "At least one allowed_root is required"
[ -n "$DELETE_DIR" ] || fail "delete_dir is required"

assert_allowed_path "$DELETE_DIR"
mkdir -p "$DELETE_DIR"
DELETE_DIR="$(resolve_path "$DELETE_DIR")"

if [ -n "$AUDIT_LOG" ]; then
  assert_allowed_path "$AUDIT_LOG"
fi

case "$command_name" in
  help)
    usage
    exit 0
    ;;
  write)
    require_args 1 "$@"
    target_abs="$(resolve_path "$1")"
    assert_allowed_path "$target_abs"
    assert_not_delete_tree "$target_abs"
    ensure_parent_dir "$target_abs"
    [ ! -d "$target_abs" ] || fail "Refusing to write to a directory: $target_abs"
    tmp_file="$(mktemp "$(dirname "$target_abs")/.safe-write.XXXXXX")" || fail "Cannot create temporary file"
    if ! cat > "$tmp_file"; then
      rm -f "$tmp_file"
      fail "Failed to read stdin for write"
    fi
    mv "$tmp_file" "$target_abs"
    log_action "write" "$target_abs" "-"
    ;;
  touch)
    require_args 1 "$@"
    target_abs="$(resolve_path "$1")"
    assert_allowed_path "$target_abs"
    assert_not_delete_tree "$target_abs"
    ensure_parent_dir "$target_abs"
    [ ! -d "$target_abs" ] || fail "Refusing to touch a directory: $target_abs"
    [ ! -e "$target_abs" ] || fail "Refusing to overwrite an existing path with touch: $target_abs"
    : > "$target_abs"
    log_action "touch" "$target_abs" "-"
    ;;
  mkdir)
    require_args 1 "$@"
    target_abs="$(resolve_path "$1")"
    assert_allowed_path "$target_abs"
    assert_not_delete_tree "$target_abs"
    mkdir -p "$target_abs"
    final_abs="$(resolve_path "$target_abs")"
    assert_allowed_path "$final_abs"
    assert_not_delete_tree "$final_abs"
    log_action "mkdir" "$final_abs" "-"
    ;;
  move)
    require_args 2 "$@"
    src_abs="$(resolve_path "$1")"
    dst_abs="$(resolve_path "$2")"
    [ -e "$src_abs" ] || fail "Source does not exist: $src_abs"
    assert_allowed_path "$src_abs"
    assert_allowed_path "$dst_abs"
    assert_not_delete_tree "$src_abs"
    assert_not_delete_tree "$dst_abs"
    dst_parent="$(dirname "$dst_abs")"
    [ -d "$dst_parent" ] || fail "Destination parent does not exist: $dst_parent"
    [ ! -e "$dst_abs" ] || fail "Destination already exists: $dst_abs"
    mv "$src_abs" "$dst_abs"
    log_action "move" "$src_abs" "$dst_abs"
    ;;
  delete)
    require_args 1 "$@"
    src_abs="$(resolve_path "$1")"
    [ -e "$src_abs" ] || fail "Target does not exist: $src_abs"
    assert_allowed_path "$src_abs"
    assert_not_delete_tree "$src_abs"
    find_allowed_root "$src_abs" || fail "Path is outside allowed roots: $src_abs"
    [ "$src_abs" != "$MATCHED_ROOT" ] || fail "Refusing to delete an allowed root: $src_abs"
    relative_path="${src_abs#"$MATCHED_ROOT"/}"
    root_id="$(sanitize_path_id "$MATCHED_ROOT")"
    delete_target="$DELETE_DIR/$root_id/$relative_path.deleted-$(now_stamp)-$$"
    mkdir -p "$(dirname "$delete_target")"
    mv "$src_abs" "$delete_target"
    log_action "delete" "$src_abs" "$delete_target"
    printf '%s\n' "$delete_target"
    ;;
  *)
    fail "Unknown command: $command_name"
    ;;
esac
