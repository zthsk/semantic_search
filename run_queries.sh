#!/bin/bash

# the list of models
models=("bert" "lsa" "lda")

# iterating over each line in queries.txt i.e. each query
while IFS= read -r query
do
    # run the queries over each model
    for model in "${models[@]}"
    do
        echo "Running query for model: $model"
        echo "Query: $query"
        python query.py --model "$model" --query "$query"
    done
done < "queries.txt"
