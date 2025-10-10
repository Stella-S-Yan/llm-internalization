

from prepare_data import download_amazon_data, preprocess_amazon_data, gen_sid, gen_embedding_sbert, gen_embedding_llama
from quantization import train_rqvae

def run_pipeline():
    # Download data
    download_amazon_data.download_amazon_data()
    # Preprocess data
    preprocess_amazon_data.run_preprocessing()
    # Get sbert embedding
    gen_embedding_sbert.gen_embedding()
    # Get llama embedding
    gen_embedding_llama.do_the_work()
    # Train rqvae
    train_rqvae.train()
    # Create sid
    gen_sid.gen_sid()
    


if __name__== "__main__":
    run_pipeline()
    print("Pipeline finished")