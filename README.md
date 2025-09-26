# nlp_learning
Tớ học lại NLP

# Virtual environment

3.8 for bilstm

source ~/llm_ws/.venv/bin/activate

3.10 for Phobert

source ~/llm_ws/.venv310/bin/activate

# Test bilstm
python bilstm/predict_one.py --model_dir bilstm_vn_sentiment_multiclass \
  --text "Thiết bị robot hút bụi khien toi thất vọng ồn shop phản hồi chậm."

# Test Phobert
python3 phobert/test_phobert.py --model_dir /home/dat/llm_ws/phobert_5cls_clean   --text "Thiết bị robot hút bụi thất vọng ồn shop phản hồi chậm."

# Test LLM
python3 llm/classify_csv_llm.py "sản phẩm laptop rất kinh khủng, gia công cực kinh khủng; phản hồi chậm, rất bực mình."

ollama ps

ollama stop qwen2.5:14b-instruct


Shop này làm ăn chán thật. Sản phẩm bị hỏng khi giao tới xong mình gọi cho shop để giải quyết mà không được!

Máy hoạt động bình thường, không có gì nổi bật cũng không có gì quá tệ.
