from app.main import run


def test_render_screen_simple_name():
    assert run("jane doe") == "Welcome, Jane Doe!"


def test_render_screen_apostrophe_name():
    assert run("o'connor") == "Welcome, O'Connor!"
