//! Oracle tests for `tree_causal_mask` and `TreePlan`.
//!
//! These tests document the correctness contract that the implementation
//! must satisfy. They are written *before* the fix so that the failing
//! output is the proof of the bugs described in
//! `docs/sessions/20260718-metal-model-runtime/05_KNOWN_ISSUES.md` (Task 4).
//!
//! Convention: `mask[r][c] == 1` means query position `r` is allowed to
//! attend to key position `c`. Positions outside `[0, seq_len)` are invalid.
//!
//! Tree layout (in-tree index):
//!   - root has in-tree index 0 (full-sequence position `offset`)
//!   - parent(i) = (i - 1) / width  for i >= 1
//!   - children of i: width*i + 1 .. width*i + width
//!   - `TreePlan::total_nodes()` counts 1 + width + width^2 + ... + width^depth

use tree_attention::{tree_causal_mask, TreePlan};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mask_at(mask: &[Vec<u8>], r: usize, c: usize) -> u8 {
    mask[r][c]
}

fn check_self_attention_diagonal(mask: &[Vec<u8>], seq_len: usize) {
    for i in 0..seq_len {
        assert_eq!(
            mask_at(mask, i, i),
            1,
            "self-attention at i={i} should be 1"
        );
    }
}

// ---------------------------------------------------------------------------
// (a) Self-attention: diagonal is always 1.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_self_attention_is_one_on_diagonal() {
    let cases = [
        (8usize, 2usize, 2usize, 4usize),
        (16, 4, 2, 8),
        (32, 3, 3, 4),
        (4, 1, 2, 2),
        (10, 2, 1, 6),
    ];
    for &(seq_len, w, d, off) in &cases {
        let mask = tree_causal_mask(seq_len, w, d, off);
        check_self_attention_diagonal(&mask, seq_len);
    }
}

// ---------------------------------------------------------------------------
// (b) Sibling isolation: width=2 depth=2, seq_len=8 offset=4.
//     Tree occupies positions 4..(4+7)=11 — but seq_len=8 caps at 8.
//     Positions 4=root, 5/6=root's children, 7=child-of-5 (5's only child).
//     Siblings: (5,6) under root. They MUST NOT attend to each other.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_rejects_sibling_attention_for_width_2_depth_2() {
    let seq_len = 8;
    let width = 2;
    let depth = 2;
    let offset = 4;
    let mask = tree_causal_mask(seq_len, width, depth, offset);

    // Tree indices (in-tree) 0..2 are positions 4..6.
    // In-tree: 0=root (pos 4), 1,2 = children of root (pos 5,6).
    // Siblings: 1 and 2 are NOT ancestors of each other.
    assert_eq!(mask[5][6], 0, "sibling 5 must not attend to sibling 6");
    assert_eq!(mask[6][5], 0, "sibling 6 must not attend to sibling 5");
}

// ---------------------------------------------------------------------------
// (c) Ancestor -> descendant attention is allowed (query sees its ancestors).
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_allows_ancestor_query_to_attend_descendant_key() {
    // width=2 depth=2, root at offset=4.
    // In-tree: 0=root (pos 4), 1=child0 (pos 5), 2=child1 (pos 6),
    // 3=grandchild-of-1 (pos 7).
    // Each tree node must see all its ancestors (and itself).
    let mask = tree_causal_mask(8, 2, 2, 4);

    // Child 5 (in-tree 1) sees the root (in-tree 0) and itself.
    assert_eq!(mask[5][4], 1, "child 5 must attend to root 4");
    assert_eq!(mask[5][5], 1, "child 5 must attend to itself");
    assert_eq!(mask[5][6], 0, "child 5 must not attend to sibling 6");
    // 5 cannot attend to its descendant 7 (child).
    assert_eq!(mask[5][7], 0, "child 5 must not attend to its child 7");

    // Grandchild 7 (in-tree 3) sees its parent 5 (in-tree 1), the root, and itself.
    assert_eq!(mask[7][4], 1, "grandchild 7 must attend to root 4");
    assert_eq!(mask[7][5], 1, "grandchild 7 must attend to its parent 5");
    assert_eq!(mask[7][7], 1, "grandchild 7 must attend to itself");
    assert_eq!(mask[7][6], 0, "grandchild 7 must not attend to its uncle 6");
}

// ---------------------------------------------------------------------------
// (d) Descendant -> ancestor-out-of-its-chain attention is forbidden.
//     (A query cannot attend to a tree node that is not its ancestor.)
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_rejects_descendant_attending_out_of_chain() {
    let mask = tree_causal_mask(8, 2, 2, 4);
    // Child 5 (in-tree 1) must NOT attend to its sibling 6 (in-tree 2)
    // nor to its niece/nephew (none in this tree) — only ancestors are seen.
    assert_eq!(mask[5][6], 0, "child 5 must not attend to sibling 6");
    // Grandchild 7 (in-tree 3) must not attend to its uncle 6 (in-tree 2,
    // which is NOT its ancestor).
    assert_eq!(mask[7][6], 0, "grandchild 7 must not attend to uncle 6");
    // Grandchild 7 must not attend to its aunt-like cousin (none here) —
    // sanity check: nothing in tree outside 7's chain is reachable.
}

// ---------------------------------------------------------------------------
// (e) Each tree node sees its entire ancestor chain.
//     Query at a deep descendant attends to every ancestor on the way up.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_query_at_descendant_sees_full_ancestor_chain() {
    // width=2 depth=2 offset=4:
    //   in-tree 0=root (pos 4)
    //   in-tree 1=child0 (pos 5), parent=0
    //   in-tree 2=child1 (pos 6), parent=0
    //   in-tree 3=grandchild (pos 7), parent=1
    let mask = tree_causal_mask(8, 2, 2, 4);
    // Grandchild (7) sees its parent (5) and grandparent (4).
    assert_eq!(mask[7][4], 1, "grandchild 7 must attend to root 4");
    assert_eq!(mask[7][5], 1, "grandchild 7 must attend to its parent 5");
    // Child 5 sees root 4.
    assert_eq!(mask[5][4], 1, "child 5 must attend to root 4");
}

// ---------------------------------------------------------------------------
// (f) Causal in prefix region: mask[r][c] = 1 iff r >= c, for c < tree_start.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_is_causal_in_prefix_region() {
    let mask = tree_causal_mask(12, 3, 2, 7);
    let tree_start = 7;
    // For c < tree_start, mask[r][c] == 1 iff r >= c.
    for c in 0..tree_start {
        for (r, row) in mask.iter().enumerate() {
            let expected = if r >= c { 1 } else { 0 };
            assert_eq!(
                row[c], expected,
                "prefix mask mismatch at r={r}, c={c}: expected {expected}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (g) Depth 0 -> degenerate tree of just the root at offset.
//     tree_start == tree_end + 1 == offset + 1 (one root).
//     Prefix attention remains standard causal.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_handles_depth_zero_correctly() {
    // depth=0, width=anything (width irrelevant for a single-node tree).
    let mask = tree_causal_mask(6, 4, 0, 3);

    // The single tree node sits at position 3.
    // Prefix queries (r < 3) cannot see the tree (c == 3).
    for (r, row) in mask.iter().take(3).enumerate() {
        assert_eq!(row[3], 0, "prefix query r={r} must not see tree root");
    }
    // The root attends to itself.
    assert_eq!(mask[3][3], 1);
    // Root attends to the prefix.
    let root_row = &mask[3];
    for (c, &cell) in root_row.iter().take(3).enumerate() {
        assert_eq!(cell, 1, "root must attend to prefix c={c}");
    }
    // Prefix causal is preserved.
    for (r, row) in mask.iter().take(3).enumerate() {
        for (c, &cell) in row.iter().take(3).enumerate() {
            let expected = if r >= c { 1 } else { 0 };
            assert_eq!(cell, expected, "prefix causal mismatch at r={r}, c={c}");
        }
    }
}

// ---------------------------------------------------------------------------
// (h) Tree length consistency with `TreePlan::total_nodes()`.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_consistent_with_tree_plan_total_nodes() {
    let plan = TreePlan::new(4, 2);
    let expected_tree_len = plan.total_nodes();
    assert_eq!(expected_tree_len, 21, "1 + 4 + 16 = 21 for width=4 depth=2");

    let seq_len = 64;
    let offset = 8;
    let mask = tree_causal_mask(seq_len, plan.width, plan.depth, offset);

    // The deepest descendant (in-tree index `expected_tree_len - 1`) sees
    // the root, and the root does NOT see its descendant.
    let tree_end = offset + expected_tree_len;
    assert_eq!(
        mask[tree_end - 1][offset],
        1,
        "deepest descendant at {} must attend to root",
        tree_end - 1
    );
    assert_eq!(
        mask[offset][tree_end - 1],
        0,
        "root must not attend to its descendant at {}",
        tree_end - 1
    );
    // Self-attend root.
    assert_eq!(mask[offset][offset], 1);
}

// ---------------------------------------------------------------------------
// (i) Singleton width=1 with depth=k is a chain -> standard causal mask
//     over positions [offset, offset + k].
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_singleton_width_one_is_causal() {
    let offset = 3;
    let depth = 4;
    let mask = tree_causal_mask(10, 1, depth, offset);

    // Tree: positions offset..offset+depth+1 (root + 4 descendants = 5 nodes).
    let tree_end = offset + depth + 1;
    // Causal within tree: r >= c (since the chain is purely linear).
    for (r, row) in (offset..tree_end).zip(mask[offset..tree_end].iter()) {
        for (c, &cell) in (offset..tree_end).zip(row[offset..tree_end].iter()) {
            let expected = if r >= c { 1 } else { 0 };
            assert_eq!(
                cell, expected,
                "width=1 chain should be causal at r={r}, c={c}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// (j) Prefix query must NOT see tree nodes (r < tree_start, c >= tree_start).
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_rejects_querying_beyond_tree_end() {
    let mask = tree_causal_mask(16, 3, 2, 10);
    let tree_start = 10;
    let tree_end = 10 + TreePlan::new(3, 2).total_nodes();
    // Prefix queries (r < tree_start) must not attend to tree keys.
    for (r, row) in mask.iter().enumerate().take(tree_start) {
        let upper = tree_end.min(row.len());
        for (c, &cell) in (tree_start..upper).zip(row[tree_start..upper].iter()) {
            assert_eq!(cell, 0, "prefix query r={r} must not see tree key c={c}");
        }
    }
}

// ---------------------------------------------------------------------------
// (k) Depth=1: root has w children; each child attends to root, root attends
//     to each child.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_depth_one_width_w() {
    let width = 4;
    let offset = 5;
    let seq_len = 20;
    let mask = tree_causal_mask(seq_len, width, 1, offset);

    // Children positions: offset+1 .. offset+width.
    for (child_idx, child_row) in
        ((offset + 1)..(offset + 1 + width)).zip(mask[(offset + 1)..(offset + 1 + width)].iter())
    {
        let child = child_idx;
        // Child attends to root (root is child's ancestor).
        assert_eq!(child_row[offset], 1, "child {child} must attend to root");
        // Child does NOT attend to its siblings (siblings are not ancestors).
        for (sibling_idx, sibling_cell) in ((offset + 1)..(offset + 1 + width))
            .zip(child_row[(offset + 1)..(offset + 1 + width)].iter())
        {
            let sibling = sibling_idx;
            if sibling == child {
                continue;
            }
            assert_eq!(
                *sibling_cell, 0,
                "child {child} must not attend to sibling {sibling}"
            );
        }
    }
    // The root does NOT attend to its children (children are descendants,
    // not ancestors of the root).
    let root_row = &mask[offset];
    for (child, &cell) in ((offset + 1)..(offset + 1 + width))
        .zip(root_row[(offset + 1)..(offset + 1 + width)].iter())
    {
        assert_eq!(cell, 0, "root must not attend to descendant {child}");
    }
}

// ---------------------------------------------------------------------------
// (l) Odd seq_len / non-power-of-two offsets must work without panicking
//     and produce a causally consistent mask.
// ---------------------------------------------------------------------------

#[test]
fn tree_causal_mask_handles_odd_offsets_and_sizes() {
    // Non-power-of-two seq_len and offset.
    let mask = tree_causal_mask(13, 2, 2, 5);
    let tree_start = 5;
    let tree_total = TreePlan::new(2, 2).total_nodes(); // 1+2+4 = 7
    let tree_end = tree_start + tree_total;

    // Self-attention on diagonal.
    check_self_attention_diagonal(&mask, 13);

    // Sibling isolation: positions 6 and 7 are siblings under the root (5).
    // Neither attends to the other.
    assert_eq!(mask[6][7], 0, "siblings 6,7 must not attend to each other");
    assert_eq!(mask[7][6], 0);

    // Children attend to root.
    assert_eq!(mask[6][5], 1, "child 6 must attend to root 5");
    assert_eq!(mask[7][5], 1, "child 7 must attend to root 5");

    // Root does NOT attend to children.
    assert_eq!(mask[5][6], 0, "root 5 must not attend to descendant 6");
    assert_eq!(mask[5][7], 0, "root 5 must not attend to descendant 7");

    // Tree region: every prefix query (< tree_start) sees no tree.
    for (r, row) in mask.iter().enumerate().take(tree_start) {
        let upper = tree_end.min(row.len());
        for (c, &cell) in (tree_start..upper).zip(row[tree_start..upper].iter()) {
            assert_eq!(cell, 0, "prefix r={r} must not see tree c={c}");
        }
    }
}
