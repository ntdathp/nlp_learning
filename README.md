# nlp_learning
Tớ học lại NLP


source ~/llm_ws/.venv/bin/activate
source ~/llm_ws/.venv310/bin/activate

# Test bilstm
python predict_sentiment.py --model_dir bilstm_vn_sentiment_multiclass \
    --csv test_samples.csv --out_csv pred.csv


export TF_CPP_MIN_LOG_LEVEL=2 
python predict_sentiment.py --model_dir bilstm_vn_sentiment_multiclass \
  --text "Thiết bị robot hút bụi thất vọng ồn shop phản hồi chậm."