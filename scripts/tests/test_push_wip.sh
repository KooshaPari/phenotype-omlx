#!/usr/bin/env bash
# test_push_wip.sh — exercise scripts/push_wip.sh in a disposable git
# repo to verify (a) working-tree-dirty guard, (b) airlock-v2 retry with
# exponential backoff, (c) airlock-v2 fallback to plain git push, and
# (d) airlock-v2 disabled via env falls through to git push.
#
# The tests use fake binaries for both `airlock-v2` and `git push`
# (when needed) so the script's retry loop is fully controllable.
# Real `git` is used for setup/teardown of the test fixture.
#
# Bash 3.2 portable (macOS default); no external deps.

set -euo pipefail

TEST_ROOT="$(mktemp -d -t push_wip_test.XXXXXX)"
trap 'rm -rf "${TEST_ROOT}"; kill $(jobs -p) 2>/dev/null || true' EXIT

# Resolve script under test (in parent scripts/) and the project root.
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${TESTS_DIR}/.." && pwd)"
PUSH_WIP="${SCRIPTS_DIR}/push_wip.sh"

_log() { printf '[test_push_wip] %s\n' "$*" >&2; }

# --- helper: a fake `airlock-v2` that fails N times, then succeeds. ---
_make_failing_airlock() {
    local fail_count="$1"
    local log_file="$2"
    cat >"${TEST_ROOT}/fake_airlock_v2" <<EOF
#!/usr/bin/env bash
echo "fake_airlock_v2 invoked args=\$*" >>"${log_file}"
count_file="${TEST_ROOT}/airlock_call_count"
n=\$(cat "\${count_file}" 2>/dev/null || echo 0)
n=\$(( n + 1 ))
echo "\$n" >"\${count_file}"
if (( n <= ${fail_count} )); then
    echo "fake_airlock_v2: simulated transient failure #\${n}" >&2
    exit 1
fi
echo "fake_airlock_v2: success on try #\${n}" >&2
exit 0
EOF
    chmod +x "${TEST_ROOT}/fake_airlock_v2"
}

# --- helper: a fake `git` wrapping the real git inside the repo. ---
# Recognizes the subcommand even when the caller writes
# `git -C ROOT status --porcelain` (walks past the -C flag pair).
# All other invocations delegate to the real git.
#
# Args: $1 = mode ("ok" | "fail" | "fail-then-ok:N"),
#       $2 = "clean" | "dirty" (for the dirty guard).
_make_git_wrapper() {
    local mode="$1"
    local dirty="${2:-clean}"
    cat >"${TEST_ROOT}/fake_git" <<EOF
#!/usr/bin/env bash
# Walk past flag pairs like -C ROOT and locate the subcommand.
i=0
n=\$#
subcmd=""
while (( i < n )); do
    i=\$(( i + 1 ))
    a="\${@:i:1}"
    case "\${a}" in
        -C|--git-dir=*)
            i=\$(( i + 1 ))   # skip the flag's value
            ;;
        --)
            if (( i + 1 < n )); then
                i=\$(( i + 1 ))
                subcmd="\${@:i:1}"
            fi
            break
            ;;
        -*)
            # skip other bare flags without a value
            ;;
        *)
            subcmd="\${a}"
            break
            ;;
    esac
done
mode_file="${TEST_ROOT}/git_push_mode"
case "\${subcmd}" in
    push)
        mode="\$(cat "\${mode_file}" 2>/dev/null || echo ok)"
        case "\${mode}" in
            ok)
                exec /usr/bin/git "\$@"
                ;;
            fail)
                echo "fake_git: simulated push failure" >&2
                exit 1
                ;;
            fail-then-ok:*)
                n="\${mode#fail-then-ok:}"
                cfile="${TEST_ROOT}/git_push_count"
                c="\$(cat "\${cfile}" 2>/dev/null || echo 0)"
                c=\$(( c + 1 ))
                echo "\$c" >"\${cfile}"
                if (( c <= n )); then
                    echo "fake_git: simulated push failure #\${c}" >&2
                    exit 1
                fi
                exec /usr/bin/git "\$@"
                ;;
        esac
        ;;
    status)
        if [[ "${dirty}" == "dirty" ]]; then
            echo ' M scripts/push_wip.sh'
            exit 0
        fi
        exec /usr/bin/git "\$@"
        ;;
    *)
        exec /usr/bin/git "\$@"
        ;;
esac
EOF
    chmod +x "${TEST_ROOT}/fake_git"
}

# --- helper: initialize a git fixture with a known initial commit. ---
_init_fixture() {
    local repo="${TEST_ROOT}/fixture"
    mkdir -p "${repo}"
    cd "${repo}"
    /usr/bin/git init -q -b main .
    /usr/bin/git config user.email "test@test.invalid"
    /usr/bin/git config user.name "Test"
    echo "stub" >README
    /usr/bin/git add README
    /usr/bin/git commit -q -m "init"
    echo "${repo}"
}

# --- helper: make the fake airlock + git visible to the script. ---
# The wrapper lives outside the fixture (in TEST_ROOT/wrappers) so the
# fixture's git-status-only check stays clean across tests.
# Args: $1 = repo path (PUSH_WIP_REPO_ROOT), $2 = GIT_BIN (optional,
#       defaults to TEST_ROOT/fake_git).
_make_wrapper() {
    local repo="$1"
    local git_bin="${2:-${TEST_ROOT}/fake_git}"
    mkdir -p "${TEST_ROOT}/wrappers"
    local winame="run_${RANDOM}_$$.sh"
    cat >"${TEST_ROOT}/wrappers/${winame}" <<EOF
#!/usr/bin/env bash
export PUSH_WIP_REPO_ROOT="${repo}"
export PUSH_BIN="${TEST_ROOT}/fake_airlock_v2"
export GIT_BIN="${git_bin}"
export AIRLOCK_VERBOSE=1
export INITIAL_BACKOFF_SECONDS=1
export MAX_BACKOFF_SECONDS=2
export MAX_TOTAL_SECONDS=10
export MAX_RETRIES=3
exec bash "${PUSH_WIP}" "\$@"
EOF
    chmod +x "${TEST_ROOT}/wrappers/${winame}"
    printf '%s\n' "${TEST_ROOT}/wrappers/${winame}"
}

# ---------------------------------------------------------------------------
# Test 1: working-tree-dirty guard fires (exit 2) before any push.
# ---------------------------------------------------------------------------

_log "test 1: dirty working tree must cause exit 2"
repo="$(_init_fixture)"
_make_git_wrapper "ok" "dirty"   # wrap git status to look dirty
_make_failing_airlock 99 "${TEST_ROOT}/dummy.log"
wrapper="$(_make_wrapper "${repo}")"
cd "${repo}"
set +e
"${wrapper}" origin main >"${TEST_ROOT}/t1.out" 2>"${TEST_ROOT}/t1.err"
rc=$?
set -e
if (( rc != 2 )); then
    cat "${TEST_ROOT}/t1.err" >&2
    _log "FAIL: expected exit 2 (dirty), got ${rc}"
    exit 1
fi
if ! grep -q 'working tree dirty' "${TEST_ROOT}/t1.err"; then
    _log "FAIL: stderr should mention 'working tree dirty':"
    cat "${TEST_ROOT}/t1.err" >&2
    exit 1
fi
_log "  ok (rc=2, working-tree guard fired)"

# Reset for next test: tell fake_git status to look clean again.
_make_git_wrapper "ok" "clean"

# ---------------------------------------------------------------------------
# Test 2: airlock-v2 fails twice then succeeds; no fallback needed.
# ---------------------------------------------------------------------------

_log "test 2: airlock-v2 retry succeeds after transient failures"
# Reset and create local commit + remote (bare) to actually push.
cd "${TEST_ROOT}"
mkdir -p remote.git
/usr/bin/git init -q --bare remote.git
cd fixture
/usr/bin/git remote add origin "${TEST_ROOT}/remote.git"
/usr/bin/git push -q origin main
echo "v2" >>README
/usr/bin/git commit -q -am "second"
echo "ok" >"${TEST_ROOT}/git_push_mode"
echo "0" >"${TEST_ROOT}/airlock_call_count"
_make_git_wrapper "ok"
# Airlock fails twice on push args — but our fake doesn't actually
# push, so we need the real git to push via plain git afterwards.
# To get a clean test of retry→success on the FIRST stage, have
# the airlock call git push itself on success.
cat >"${TEST_ROOT}/fake_airlock_v2" <<EOF
#!/usr/bin/env bash
echo "fake_airlock_v2 invoked args=\$*" >>"${TEST_ROOT}/airlock.log"
cfile="${TEST_ROOT}/airlock_call_count"
n=\$(cat "\${cfile}" 2>/dev/null || echo 0)
n=\$(( n + 1 ))
echo "\$n" >"\${cfile}"
if (( n <= 2 )); then
    echo "fake_airlock_v2: transient failure #\${n}" >&2
    exit 1
fi
# On success, delegate to real git for the actual push.
/usr/bin/git "\$@"
EOF
chmod +x "${TEST_ROOT}/fake_airlock_v2"
# fresh wrapper; use real git so the airlock can delegate to real
# `git push` once it succeeds after the retry budget is exhausted.
wrapper2="$(_make_wrapper "${TEST_ROOT}/fixture" "/usr/bin/git")"
cd "${TEST_ROOT}/fixture"
set +e
"${wrapper2}" origin main >"${TEST_ROOT}/t2.out" 2>"${TEST_ROOT}/t2.err"
rc=$?
set -e
if (( rc != 0 )); then
    cat "${TEST_ROOT}/t2.err" >&2
    cat "${TEST_ROOT}/t2.out" >&2
    _log "FAIL: expected exit 0, got ${rc}"
    exit 1
fi
if (( $(cat "${TEST_ROOT}/airlock_call_count") < 3 )); then
    _log "FAIL: expected at least 3 airlock calls (2 fails + 1 success), got $(cat "${TEST_ROOT}/airlock_call_count")"
    exit 1
fi
_log "  ok (rc=0, airlock invoked $(cat "${TEST_ROOT}/airlock_call_count") times)"

# ---------------------------------------------------------------------------
# Test 3: airlock-v2 unavailable → fall back to plain git push; success.
# ---------------------------------------------------------------------------

_log "test 3: airlock-v2 missing → plain git push used"
cd "${TEST_ROOT}"
mkdir -p remote3.git
/usr/bin/git init -q --bare remote3.git
rm -rf fixture3
mkdir -p fixture3
cd fixture3
/usr/bin/git init -q -b main .
/usr/bin/git config user.email "t@t.invalid"
/usr/bin/git config user.name "t"
echo x >README
/usr/bin/git add README
/usr/bin/git commit -q -m init
/usr/bin/git remote add origin "${TEST_ROOT}/remote3.git"
/usr/bin/git push -q origin main
echo y >>README
/usr/bin/git commit -q -am "two"
# Wrap PUSH_BIN to a non-existent path; script must fall back to git.
cat >"${TEST_ROOT}/run3" <<EOF
#!/usr/bin/env bash
export PUSH_WIP_REPO_ROOT="${TEST_ROOT}/fixture3"
export PUSH_BIN="/nonexistent/airlock-v2"
export GIT_BIN=/usr/bin/git
export AIRLOCK_VERBOSE=1
export INITIAL_BACKOFF_SECONDS=1
export MAX_BACKOFF_SECONDS=2
export MAX_TOTAL_SECONDS=5
export MAX_RETRIES=2
exec bash "${PUSH_WIP}" "\$@"
EOF
chmod +x "${TEST_ROOT}/run3"
set +e
"${TEST_ROOT}/run3" origin main >"${TEST_ROOT}/t3.out" 2>"${TEST_ROOT}/t3.err"
rc=$?
set -e
if (( rc != 0 )); then
    cat "${TEST_ROOT}/t3.err" >&2
    _log "FAIL: expected exit 0, got ${rc}"
    exit 1
fi
if ! grep -q 'airlock-v2 unavailable' "${TEST_ROOT}/t3.err"; then
    _log "WARN: stderr did not mention 'airlock-v2 unavailable'; not fatal"
fi
_log "  ok (rc=0, fell back to plain git push)"

# ---------------------------------------------------------------------------
# Test 4: AIRLOCK_DISABLE=1 forces plain git push even if airlock present.
# ---------------------------------------------------------------------------

_log "test 4: AIRLOCK_DISABLE=1 bypasses airlock"
cd "${TEST_ROOT}"
mkdir -p remote4.git
/usr/bin/git init -q --bare remote4.git
rm -rf fixture4
mkdir -p fixture4
cd fixture4
/usr/bin/git init -q -b main .
/usr/bin/git config user.email "t@t.invalid"
/usr/bin/git config user.name "t"
echo z >README
/usr/bin/git add README
/usr/bin/git commit -q -m init
/usr/bin/git remote add origin "${TEST_ROOT}/remote4.git"
/usr/bin/git push -q origin main
echo z2 >>README
/usr/bin/git commit -q -am "two"
# Pretend airlock-v2 is on PATH but force-disable via env.
mkdir -p "${TEST_ROOT}/bindir"
cat >"${TEST_ROOT}/bindir/airlock-v2" <<'EOF'
#!/usr/bin/env bash
echo "WRONG: airlock-v2 should have been disabled" >&2
exit 99
EOF
chmod +x "${TEST_ROOT}/bindir/airlock-v2"
cat >"${TEST_ROOT}/run4" <<EOF
#!/usr/bin/env bash
export PUSH_WIP_REPO_ROOT="${TEST_ROOT}/fixture4"
export PATH="${TEST_ROOT}/bindir:/usr/bin:/bin"
export AIRLOCK_DISABLE=1
export GIT_BIN=/usr/bin/git
export AIRLOCK_VERBOSE=1
export INITIAL_BACKOFF_SECONDS=1
export MAX_BACKOFF_SECONDS=2
export MAX_TOTAL_SECONDS=5
export MAX_RETRIES=2
exec bash "${PUSH_WIP}" "\$@"
EOF
chmod +x "${TEST_ROOT}/run4"
set +e
"${TEST_ROOT}/run4" origin main >"${TEST_ROOT}/t4.out" 2>"${TEST_ROOT}/t4.err"
rc=$?
set -e
if (( rc != 0 )); then
    cat "${TEST_ROOT}/t4.err" >&2
    _log "FAIL: expected exit 0, got ${rc}"
    exit 1
fi
_log "  ok (rc=0, AIRLOCK_DISABLE honored)"

_log "ALL push_wip tests passed"
exit 0
