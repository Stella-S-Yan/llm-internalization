

from prepare_data import download_amazon_data, preprocess_amazon_data, gen_sid, gen_embedding_sbert, add_seq_user_data, gen_embedding_llama
from quantization import train_rqvae
from use_all_data import fixed_grain_dataset

def run_pipeline():
    # Download data
    download_amazon_data.download_amazon_data()
    # Preprocess data
    preprocess_amazon_data.run_preprocessing()
    # Get sbert embedding
    gen_embedding_sbert.gen_embedding()
    # Train rqvae
    train_rqvae.train()
    # Create sid
    gen_sid.gen_sid()
    # Gen llama embedding
    gen_embedding_llama.gen_embedding()
    # Add User id
    add_seq_user_data.main()
    # Make the dataset
    fixed_grain_dataset.generate_fixed_split_data()
    


if __name__== "__main__":
    run_pipeline()
    print("Pipeline finished")