//! The headline M2 end-to-end test: drive a mock agent conversation through
//! the LCM store and assert that auto-archive fires + round-trip works.

use polymorph::{db, lcm};

/// Build a deterministic ~target-token filler string via the M1 tokenizer.
fn filler(target_tokens: usize) -> String {
    let unit = "lorem ipsum dolor sit amet consectetur adipiscing elit ";
    let mut s = String::new();
    loop {
        let (ids, _) = polymorph::tokenizer::token_spans(&s).unwrap();
        if ids.len() >= target_tokens {
            return s;
        }
        s.push_str(unit);
    }
}

#[test]
fn mock_agent_conversation_loop() {
    let pool = db::test_pool().unwrap();
    let conv = "agent-conv";
    let threshold: u64 = 1_000;
    let chunk = filler(100);

    let mut archived_nodes: Vec<String> = Vec::new();
    for _ in 0..20 {
        lcm::append(conv, "user", &chunk, &pool).unwrap();
        if let Some(node_id) = lcm::maybe_archive(conv, threshold, &pool).unwrap() {
            archived_nodes.push(node_id);
        }
    }

    assert!(
        !archived_nodes.is_empty(),
        "at least one archive node must fire over 20 turns @ ~100 tok each"
    );

    let active_after = lcm::active_token_count(conv, &pool).unwrap();
    assert!(
        active_after <= threshold * 2,
        "active count {active_after} should be bounded after archives ran (threshold={threshold})"
    );

    // Round-trip: every archived node round-trips verbatim.
    for node_id in &archived_nodes {
        let meta = lcm::describe(node_id, &pool).unwrap();
        assert_eq!(meta.depth, 0);
        assert!(meta.child_count >= 1);
        let rows = lcm::expand(node_id, &pool).unwrap();
        assert_eq!(rows.len(), meta.child_count);
        for row in &rows {
            assert_eq!(row.content, chunk);
            assert_eq!(row.role, "user");
            assert!(row.tokens > 0);
        }
    }
}

#[test]
fn archive_preserves_turn_index_ordering() {
    let pool = db::test_pool().unwrap();
    let conv = "ordered-conv";
    let chunk = filler(120);

    for _ in 0..10 {
        lcm::append(conv, "user", &chunk, &pool).unwrap();
    }
    let node_id = lcm::maybe_archive(conv, 200, &pool).unwrap().unwrap();
    let rows = lcm::expand(&node_id, &pool).unwrap();
    assert!(rows.len() >= 2);
    for w in rows.windows(2) {
        assert!(
            w[0].turn_index < w[1].turn_index,
            "archived rows must be returned in turn-index order"
        );
    }
}

#[test]
fn multiple_conversations_do_not_interfere() {
    let pool = db::test_pool().unwrap();
    let chunk = filler(150);

    for _ in 0..6 {
        lcm::append("conv-a", "user", &chunk, &pool).unwrap();
        lcm::append("conv-b", "user", &chunk, &pool).unwrap();
    }

    let node_a = lcm::maybe_archive("conv-a", 300, &pool).unwrap().unwrap();
    let node_b = lcm::maybe_archive("conv-b", 300, &pool).unwrap().unwrap();
    assert_ne!(node_a, node_b);

    let meta_a = lcm::describe(&node_a, &pool).unwrap();
    let meta_b = lcm::describe(&node_b, &pool).unwrap();
    assert_eq!(meta_a.conversation_id, "conv-a");
    assert_eq!(meta_b.conversation_id, "conv-b");
}
