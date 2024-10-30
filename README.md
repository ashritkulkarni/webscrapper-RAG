Thanks for the information. Here's a draft README for your repository:

---

# Webscrapper + RAG Framework

This project is a webscrapper combined with a Retrieval-Augmented Generation (RAG) framework. It uses a custom script to scrape all the web links of a website and then builds a RAG model using LLaMA 2 and BGE word embeddings. The vector database used is Pinecone.

## Features

- **Web Scraping**: Custom script to scrape all web links of a specified website.
- **RAG Model**: Utilizes LLaMA 2 and BGE word embeddings for building the RAG model.
- **Vector Database**: Integration with Pinecone for vector storage and retrieval.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/ashritkulkarni/webscrapper-RAG.git
   cd webscrapper-RAG
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up Pinecone:
   - Sign up at [Pinecone](https://www.pinecone.io/) and get your API key.
   - Configure your Pinecone environment with the API key.

## Usage

1. Run the webscrapper to scrape web links:
   ```sh
   python main.py --scrape
   ```

2. Build the RAG model using the scraped data:
   ```sh
   python main.py --build-rag
   ```

3. Query the RAG model:
   ```sh
   python main.py --query "Your query here"
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---

Feel free to modify this as needed!
