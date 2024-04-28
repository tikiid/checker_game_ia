from checker_model import CheckerModel
import tqdm
import time

number_games_to_test = 50
wins_player_1 = 0
wins_player_2 = 0

mean_partie_time = []

t1, t2 = 0, 0

depth_player_1 = 15
depth_player_2 = 5

# print(f"nombre de piece : {checker_model_object.number_of_piece}")


for _ in tqdm.tqdm(range(number_games_to_test)):
	checker_model_object = CheckerModel()

	game_time = 0

	while True:
		t1 = time.time()
		# checker_model_object.ia_move(model="minimax", depth_minimax=depth_minimax_player_1, to_maximise=True)
		# checker_model_object.ia_move(model="random")
		checker_model_object.ia_move(model="montecarlo", number_of_games=30,depth=depth_player_1, agent_turn=-1)
		game_state = checker_model_object.check_game_state()
		if game_state == "draw_game":
			break
		
		elif game_state == 1:
			wins_player_1 +=1
			break


		checker_model_object.ia_move(model="random")
		# checker_model_object.ia_move(model="montecarlo", number_of_games=number_games_to_test, player=1)
		# checker_model_object.ia_move(model="minimax", depth_minimax=depth_minimax_player_2, to_maximise=False)
		game_state = checker_model_object.check_game_state()
		if game_state == "draw_game":
			break
		
		elif game_state == -1:
			wins_player_2 +=1
			break

		t2 = time.time()
		game_time += t2 - t1 

	mean_partie_time.append(game_time)
	t1, t2 =  0, 0




# print

print(f"nombre de pieces : {checker_model_object.number_of_piece}")
print(f"player 1  wins {wins_player_1}")
print(f"player 2  wins {wins_player_2}")
print(f"draws {number_games_to_test - wins_player_1 - wins_player_2}")
print(f"average game time: {sum(mean_partie_time) / len(mean_partie_time) }")
print(f"game time: {sum(mean_partie_time)}")

if depth_player_1:
	print()

if depth_player_2:
	print()
