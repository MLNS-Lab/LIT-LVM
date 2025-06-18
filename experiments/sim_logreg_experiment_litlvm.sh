#!/bin/bash

gamma_values=(0.01 0.1 1 10)
sigma_values=(1)
p_values=(100 200)

# Define the session name
session_name="experiment_litlvm"

# Create a new tmux session in detached mode
tmux new-session -d -s "$session_name"

# Window counter to track the current window index
window_counter=0

for gamma in "${gamma_values[@]}"; do
    for sigma in "${sigma_values[@]}"; do
        for p in "${p_values[@]}"; do

        if (( $(echo "$sigma == 0" | bc -l) )); then
            sparsity=0
        else
            sparsity=1
        fi

            if [ "$window_counter" -gt 0 ]; then
                tmux new-window -t "$session_name"
            fi
            
            # Run the command in the current window
            tmux send-keys -t "$session_name:$window_counter" "python sim_logreg_experiment_litlvm.py --d=2 --model=litlvm --sparsity=1 --interaction=1 --noise=1 --V_noise=0.01 --sigma=${sigma} --gamma=${gamma} --p=${p}" Enter
                        
            # Increment the window counter
            window_counter=$((window_counter + 1))
        done
    done
done

# Optional: Attach to the session
tmux attach-session -t "$session_name"
####################################################
####################################################
