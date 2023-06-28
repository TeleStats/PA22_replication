#!/usr/bin/env bash
politics_path="/home/agirbau/work/PA22_replication"
#python_politics="$politics_path/venv/bin/python"
python_politics="/home/agirbau/work/politics/venv/bin/python"


export PYTHONPATH="$PYTHONPATH":"$python_politics"

cd "$politics_path"
# Models path
# For CNNW, FOXNEWSW, MSNBCW
channels=("CNNW" "FOXNEWSW" "MSNBCW")
models_path="$politics_path/data/faces_us_individuals"

# Basic settings
mode="train"
feats="resnetv1"  # resnetv1, facenet
detector=("mtcnn")

# Modifiers
from_date="2000_01_01"
to_date="2100_01_01"
modifier="sf"  # Stanford paper

# Classification step
mod_feat=("fcg_average_vote" "fcgNT_average_vote")  # Tracking and no tracking for MTCNN-facenet and resnetv1

for chan in "${channels[@]}"; do
  for det in "${detector[@]}"; do
    # 3. Run the classifier/s (can be done in parallel processes)
    for mod in "${mod_feat[@]}"; do

      echo "Running classification for $mode from $from_date to $to_date with $det-$feats-$mod..."
      "$python_politics" "$politics_path"/src/face_classifier.py "$mode" "$chan" --models_path "$models_path" --detector "$det" --feats "$feats" --mod_feat "$mod" --from_date "$from_date" --to_date "$to_date" --extract_embs
      echo "...done!"
      wait

      # 4. Run evaluation of the results (in demo is also used for generating the file to analyse)
      # Save results
      results_path="$politics_path"/data/results/"$chan"/"$mode"/results
      mkdir "$results_path"

      # Academia-oriented evaluation (do this only for "train")
      if [ "$mode" == "train" ]; then
        echo "Evaluating sytem for $det-$feats-$mod-sf..."
        res_file="$results_path"/"$det"-"$feats"-"$mod"-sf.txt
        "$python_politics" "$politics_path"/src/metrics.py "$mode" "$chan" --models_path "$models_path" --detector "$det" --feats "$feats" --mod_feat "$mod" --from_date "$from_date" --to_date "$to_date" --modifier "$modifier" > "$res_file"
        echo "...done!"
        wait
      fi
    done
  done
done