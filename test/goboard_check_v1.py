'''
def check_sample_v1(case: TestCase, Board: Type[GoBoard]):
    with SgfDataBase() as data:
        games = data.sample(5)
        for sgf_file in games:
            b = Board(19,sgf_file.first_player)
            b.setup_stones(*sgf_file.setup_stones)
            for pos in sgf_file.sequence:
                b.play(pos)
                for string in b.get_strings():
                    for stone in string.stones:
                        case.assertEqual(b.get_string(stone), string)
'''