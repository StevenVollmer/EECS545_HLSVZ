from app.graph.cycle import has_cycle


def test_empty_graph_has_no_cycle():
    assert has_cycle({}) is False


def test_simple_self_loop_is_a_cycle():
    assert has_cycle({"a": ["a"]}) is True


def test_linear_chain_has_no_cycle():
    # a -> b -> c, single connected chain explored starting from 'a'.
    assert has_cycle({"a": ["b"], "b": ["c"], "c": []}) is False


def test_back_edge_in_single_dfs_is_a_cycle():
    # a -> b -> c -> a, whole cycle found during the same DFS from 'a'.
    assert has_cycle({"a": ["b"], "b": ["c"], "c": ["a"]}) is True
