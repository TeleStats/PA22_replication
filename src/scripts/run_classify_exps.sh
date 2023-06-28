#!/usr/bin/env bash
politics_path="/home/agirbau/work/PA22_replication"
#python_politics="$politics_path/venv/bin/python"
python_politics="/home/agirbau/work/politics/venv/bin/python"


export PYTHONPATH="$PYTHONPATH":"$python_politics"

cd "$politics_path"
# Models path
# For News7 and Hodo station
channels=("news7-lv" "hodost-lv")
models_path="$politics_path/data/faces_politicians"
# For CNNW, FOXNEWSW, MSNBCW
#channels=("CNNW" "FOXNEWSW" "MSNBCW")
#channels=("CNNW")
#models_path="$politics_path/data/faces_us_individuals"

# Basic settings
mode="train"
feats="resnetv1"
detector=("yolo")
#detector=("yolo" "dfsd" "mtcnn")

# Modifiers
from_date="2000_01_01"
to_date="2100_01_01"
modifier="sf"  # Stanford paper

# Classification step
#mod_feat=("fcg_average_vote")
#mod_feat=("fcg_average_vote" "fcgNT_average_vote" "fcg_average_centroid" "knn_3")

# Experiments for PA rebuttal on different voting thresholds
perc_votes=("0.1" "0.15" "0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5" "0.55" "0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.9" "0.95" "0.99")
mod_feat=("fcg_average_vote_01" "fcg_average_vote_015" "fcg_average_vote_02" "fcg_average_vote_025" "fcg_average_vote_03" "fcg_average_vote_035" "fcg_average_vote_04" "fcg_average_vote_045" "fcg_average_vote_05" "fcg_average_vote_055" "fcg_average_vote_06" "fcg_average_vote_065" "fcg_average_vote_07" "fcg_average_vote_075" "fcg_average_vote_08" "fcg_average_vote_085" "fcg_average_vote_09" "fcg_average_vote_095" "fcg_average_vote_099")

# 1.Extract detected faces and features from models (specify whatever channel)
# Shouldn't be any need to uncomment this
#for det in "${detector[@]}"; do
#  echo "Extracting detections and features from models with $det-$feats..."
#  "$python_politics" "$politics_path"/src/scripts/extract_model_embeddings.py "$mode" news7-lv --models_path "$models_path" --detector "$det" --feats "$feats"
#  wait
#  echo "...done!"
#done

for chan in "${channels[@]}"; do
  for det in "${detector[@]}"; do
    # 3. Run the classifier/s (can be done in parallel processes)
#    for mod in "${mod_feat[@]}"; do
    for i in "${!mod_feat[@]}"; do
      mod=${mod_feat[i]}
      perc_vote=${perc_votes[i]}

      echo "Running classification for $mode from $from_date to $to_date with $det-$feats-$mod..."
#      "$python_politics" "$politics_path"/src/face_classifier_politics.py "$mode" "$chan" --models_path "$models_path" --detector "$det" --feats "$feats" --mod_feat "$mod" --from_date "$from_date" --to_date "$to_date" --extract_embs
      "$python_politics" "$politics_path"/src/face_classifier.py "$mode" "$chan" --models_path "$models_path" --detector "$det" --feats "$feats" --mod_feat "$mod" --from_date "$from_date" --to_date "$to_date" --extract_embs
      echo "...done!"
      wait

      # 4. Run evaluation of the results (in demo is also used for generating the file to analyse)
      # Save results
      results_path="$politics_path"/data/results/"$chan"/"$mode"/results
      mkdir "$results_path"

      # Run metrics from 2 perspectives, application-oriented (with noisy elements) and academia-oriented only annotated frames
      echo "Evaluating sytem for $det-$feats-$mod..."
      # Application-oriented evaluation
      res_file="$results_path"/"$det"-"$feats"-"$mod".txt
#      "$python_politics" "$politics_path"/src/metrics.py "$mode" "$chan" --models_path "$models_path" --detector "$det" --feats "$feats" --from_date "$from_date" --mod_feat "$mod" > "$res_file"
      "$python_politics" "$politics_path"/src/metrics.py "$mode" "$chan" --models_path "$models_path" --detector "$det" --feats "$feats" --mod_feat "$mod" > "$res_file"
      echo "...done!"
      wait
      # Academia-oriented evaluation (do this only for "train")
      if [ "$mode" == "train" ]; then
        echo "Evaluating sytem for $det-$feats-$mod-sf..."
        res_file="$results_path"/"$det"-"$feats"-"$mod"-sf.txt
        "$python_politics" "$politics_path"/src/metrics.py "$mode" "$chan" --models_path "$models_path" --detector "$det" --feats "$feats" --mod_feat "$mod" --modifier "$modifier" > "$res_file"
        echo "...done!"
        wait
      fi
    done
  done
done