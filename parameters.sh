gpulaunch2 --runs=5 basic.py -num_generations 1500 -season_len 20 
gpulaunch2 --runs=5 basic.py -num_generations 3750 -season_len 50 
gpulaunch2 --runs=5 basic.py -num_generations 7500 -season_len 100 
gpulaunch2 --runs=5 basic.py -num_generations 22500 -season_len 300 
gpulaunch2 --runs=5 basic.py -num_generations 30000 -season_len 400 
gpulaunch2 --runs=5 basic.py -num_generations 37500 -season_len 500 

gpulaunch2 --runs=10 basic.py -num_generations 3750 -season_len 50 -truncation_prop 0.1
gpulaunch2 --runs=10 basic.py -num_generations 22500 -season_len 300 -truncation_prop 0.1
gpulaunch2 --runs=10 basic.py -num_generations 37500 -season_len 500 -truncation_prop 0.1
gpulaunch2 --runs=10 basic.py -num_generations 3750 -season_len 50 -truncation_prop 0.05
gpulaunch2 --runs=10 basic.py -num_generations 22500 -season_len 300 -truncation_prop 0.05
gpulaunch2 --runs=10 basic.py -num_generations 37500 -season_len 500 -truncation_prop 0.05
gpulaunch2 --runs=10 basic.py -num_generations 3750 -season_len 50 -mut_rate 0.05
gpulaunch2 --runs=10 basic.py -num_generations 22500 -season_len 300 -mut_rate 0.05
gpulaunch2 --runs=10 basic.py -num_generations 37500 -season_len 500 -mut_rate 0.05
gpulaunch2 --runs=10 basic.py -num_generations 3750 -season_len 50 -mut_rate 0.2
gpulaunch2 --runs=10 basic.py -num_generations 22500 -season_len 300 -mut_rate 0.2
gpulaunch2 --runs=10 basic.py -num_generations 37500 -season_len 500 -mut_rate 0.2