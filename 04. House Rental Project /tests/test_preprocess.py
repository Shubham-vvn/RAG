from src.preprocess import parse_floor_string

def test_parse_floor_string_standard():
    assert parse_floor_string("2 out of 5") == (2, 5)
    assert parse_floor_string("1 out of 3") == (1, 3)

def test_parse_floor_string_ground():
    assert parse_floor_string("Ground out of 2") == (0, 2)
    assert parse_floor_string("Ground out of 4") == (0, 4)
    assert parse_floor_string("Ground") == (0, 1)

def test_parse_floor_string_basement():
    assert parse_floor_string("Basement out of 4") == (-1, 4)
    assert parse_floor_string("Lower Basement out of 3") == (-2, 3)
    assert parse_floor_string("Basement") == (-1, 1)

def test_parse_floor_string_digits():
    assert parse_floor_string("3") == (3, 3)
    assert parse_floor_string("10") == (10, 10)

def test_parse_floor_string_invalid_logical():
    # Floor is greater than total floors, should swap/enforce total floors = floor
    assert parse_floor_string("5 out of 3") == (5, 5)

def test_parse_floor_string_empty():
    assert parse_floor_string(None) == (0, 1)
    assert parse_floor_string("") == (1, 1)
