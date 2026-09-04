from floop_agents.real_tasks import smoke_tasks


def test_smoke_task_families_present():
    tasks = smoke_tasks()
    assert {task.family for task in tasks} == {"solvable", "unsatisfiable", "open_ended"}


def test_objective_scoring():
    tasks = {task.task_id: task for task in smoke_tasks()}
    assert tasks["solvable_arithmetic"].score("869") == 1.0
    assert tasks["solvable_arithmetic"].score("870") == 0.0
    assert tasks["unsat_modular"].is_success("NO_SOLUTION")
    assert not tasks["unsat_modular"].is_success("1")


def test_open_ended_checklist():
    task = {task.task_id: task for task in smoke_tasks()}["open_ended_microabstract"]
    good = "An agent needs termination evidence to show that stopping preserves quality while avoiding needless computation."
    bad = "Make it better."
    assert task.is_success(good)
    assert task.score(bad) < 1.0
