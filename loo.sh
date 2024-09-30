#!/bin/bash

# Set the number of iterations
num_iterations=10
drug_range=12

# Loop through the number of iterations
for ((j=1; j<=num_iterations; j++)); do
    # Generate a random integer for i between 1 and 10000 (you can adjust this range)
    i=$((RANDOM % 10000 + 1))  # This generates a random integer between 1 and 10000

    # Loop through the values of drug from 1 to 12
    for ((drug=1; drug<=drug_range; drug++)); do
        # Run the Python script with the generated i and current drug value
        python scripts/main_torch.py -config=configs/Example.leave_one_drug_out.json -i=$i -drug=$drug
    done
done
