{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vYTttCUNikSh"
      },
      "outputs": [],
      "source": [
        "!pip install faiss-cpu\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mM9wa_ToAf_D",
        "outputId": "3ccf506a-03cc-4768-be88-03b6c8f8dda0"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting tabula-py\n",
            "  Downloading tabula_py-2.10.0-py3-none-any.whl.metadata (7.6 kB)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)\n",
            "Requirement already satisfied: sentence-transformers in /usr/local/lib/python3.12/dist-packages (5.3.0)\n",
            "Requirement already satisfied: google-generativeai in /usr/local/lib/python3.12/dist-packages (0.8.6)\n",
            "Requirement already satisfied: numpy>1.24.4 in /usr/local/lib/python3.12/dist-packages (from tabula-py) (2.0.2)\n",
            "Requirement already satisfied: distro in /usr/local/lib/python3.12/dist-packages (from tabula-py) (1.9.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.3)\n",
            "Requirement already satisfied: transformers<6.0.0,>=4.41.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (5.0.0)\n",
            "Requirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.7.1)\n",
            "Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (2.10.0+cpu)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.6.1)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.16.3)\n",
            "Requirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.15.0)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.67.3)\n",
            "Requirement already satisfied: google-ai-generativelanguage==0.6.15 in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (0.6.15)\n",
            "Requirement already satisfied: google-api-core in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.30.0)\n",
            "Requirement already satisfied: google-api-python-client in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.192.0)\n",
            "Requirement already satisfied: google-auth>=2.15.0 in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.47.0)\n",
            "Requirement already satisfied: protobuf in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (5.29.6)\n",
            "Requirement already satisfied: pydantic in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.12.3)\n",
            "Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in /usr/local/lib/python3.12/dist-packages (from google-ai-generativelanguage==0.6.15->google-generativeai) (1.27.1)\n",
            "Requirement already satisfied: googleapis-common-protos<2.0.0,>=1.56.3 in /usr/local/lib/python3.12/dist-packages (from google-api-core->google-generativeai) (1.73.0)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.20.0 in /usr/local/lib/python3.12/dist-packages (from google-api-core->google-generativeai) (2.32.4)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (0.4.2)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (4.9.1)\n",
            "Requirement already satisfied: filelock>=3.10.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.25.2)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2025.3.0)\n",
            "Requirement already satisfied: hf-xet<2.0.0,>=1.4.2 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.4.2)\n",
            "Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (0.28.1)\n",
            "Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (26.0)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.3)\n",
            "Requirement already satisfied: typer in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (0.24.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.6.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (2025.11.3)\n",
            "Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.22.2)\n",
            "Requirement already satisfied: typer-slim in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.24.0)\n",
            "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.7.0)\n",
            "Requirement already satisfied: httplib2<1.0.0,>=0.19.0 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (0.31.2)\n",
            "Requirement already satisfied: google-auth-httplib2<1.0.0,>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (0.3.0)\n",
            "Requirement already satisfied: uritemplate<5,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (4.2.0)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic->google-generativeai) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic->google-generativeai) (2.41.4)\n",
            "Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from pydantic->google-generativeai) (0.4.2)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (1.5.3)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (3.6.0)\n",
            "Requirement already satisfied: grpcio<2.0.0,>=1.33.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai) (1.78.0)\n",
            "Requirement already satisfied: grpcio-status<2.0.0,>=1.33.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai) (1.71.2)\n",
            "Requirement already satisfied: pyparsing<4,>=3.1 in /usr/local/lib/python3.12/dist-packages (from httplib2<1.0.0,>=0.19.0->google-api-python-client->google-generativeai) (3.3.2)\n",
            "Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (4.12.1)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (2026.2.25)\n",
            "Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (1.0.9)\n",
            "Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (3.11)\n",
            "Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub>=0.20.0->sentence-transformers) (0.16.0)\n",
            "Requirement already satisfied: pyasn1<0.7.0,>=0.6.1 in /usr/local/lib/python3.12/dist-packages (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai) (0.6.3)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.20.0->google-api-core->google-generativeai) (3.4.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.20.0->google-api-core->google-generativeai) (2.5.0)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.3)\n",
            "Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.20.0->sentence-transformers) (8.3.1)\n",
            "Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.20.0->sentence-transformers) (1.5.4)\n",
            "Requirement already satisfied: rich>=12.3.0 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.20.0->sentence-transformers) (13.9.4)\n",
            "Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer->huggingface-hub>=0.20.0->sentence-transformers) (0.0.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=0.20.0->sentence-transformers) (4.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=12.3.0->typer->huggingface-hub>=0.20.0->sentence-transformers) (2.19.2)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=12.3.0->typer->huggingface-hub>=0.20.0->sentence-transformers) (0.1.2)\n",
            "Downloading tabula_py-2.10.0-py3-none-any.whl (12.0 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m12.0/12.0 MB\u001b[0m \u001b[31m40.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: tabula-py\n",
            "Successfully installed tabula-py-2.10.0\n"
          ]
        }
      ],
      "source": [
        "!pip install tabula-py pandas sentence-transformers google-generativeai\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "O5MVALbwBaue"
      },
      "outputs": [],
      "source": [
        "path='/Orders.xlsx'\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "T0Dt93TTBaxH"
      },
      "outputs": [],
      "source": [
        "import tabula\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import faiss\n",
        "from sentence_transformers import SentenceTransformer\n",
        "import google.generativeai as genai\n",
        "from google.colab import userdata"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gQfxYyaUBazx"
      },
      "outputs": [],
      "source": [
        "Gemini_API_Key=userdata.get('Gemini_API_Key')\n",
        "if Gemini_API_Key is None:\n",
        "  raise ValueError(\"No Gemini_API_Key is found\")\n",
        "genai.configure(api_key=Gemini_API_Key)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "u5ss7Xj-Ba2G",
        "outputId": "9480048c-60d4-4ea9-e5fd-0498a320abe0"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "    order_id     order_status              customer order_date  \\\n",
            "0          3  Order\\rFinished  Muhammed Mac\\rIntyre 2010-10-13   \n",
            "1        293  Order\\rFinished          Barry French 2012-10-01   \n",
            "2        483  Order\\rFinished         Clay Rozendal 2011-07-10   \n",
            "3        515  Order\\rFinished        Carlos Soltero 2010-08-28   \n",
            "4        613  Order\\rFinished          Carl Jackson 2011-06-17   \n",
            "5        643  Order\\rFinished        Monica Federle 2011-03-24   \n",
            "6        678  Order\\rReturned       Dorothy Badders 2010-02-26   \n",
            "7        807  Order\\rFinished       Neola Schneider 2010-11-23   \n",
            "8        868  Order\\rFinished           Carlos Daly 2012-06-08   \n",
            "9        933  Order\\rFinished         Claudia Miner 2012-08-04   \n",
            "10       995  Order\\rFinished       Neola Schneider 2011-05-30   \n",
            "11       998  Order\\rFinished      Allen Rosenblatt 2009-11-25   \n",
            "12      1154  Order\\rFinished       Sylvia Foulston 2012-02-14   \n",
            "13      1344  Order\\rFinished           Jim Radford 2012-04-15   \n",
            "14      1412  Order\\rFinished        Carlos Soltero 2010-03-12   \n",
            "15      1539  Order\\rFinished           Carl Ludwig 2011-03-09   \n",
            "16      1540  Order\\rFinished            Don Miller 2012-08-04   \n",
            "17      1702  Order\\rFinished          Annie Cyprus 2011-05-06   \n",
            "18      1761  Order\\rFinished           Carl Ludwig 2010-12-23   \n",
            "19      1792  Order\\rFinished        Carlos Soltero 2010-11-08   \n",
            "\n",
            "    order_quantity     sales  \n",
            "0                6    523080  \n",
            "1               49  20246040  \n",
            "2               30   9931519  \n",
            "3               19    788540  \n",
            "4               12    187080  \n",
            "5               21   5563640  \n",
            "6               44    456820  \n",
            "7               45    393700  \n",
            "8               32   1433680  \n",
            "9               15    161220  \n",
            "10              46   3630980  \n",
            "11              16    496520  \n",
            "12              44   8924460  \n",
            "13              15   1669808  \n",
            "14              13    118060  \n",
            "15              33   1023660  \n",
            "16              30    161800  \n",
            "17              23    134480  \n",
            "18              25  24056460  \n",
            "19              28    740960  \n"
          ]
        }
      ],
      "source": [
        "\n",
        "df=pd.read_excel(path)\n",
        "print(df)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 394
        },
        "id": "ldmplSZ-Ba5e",
        "outputId": "c34570f9-46cf-490a-be8b-8a65dfa98d17"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "summary": "{\n  \"name\": \"meta_data\",\n  \"rows\": 11,\n  \"fields\": [\n    {\n      \"column\": \"order_id\",\n      \"properties\": {\n        \"dtype\": \"date\",\n        \"min\": \"1970-01-01 00:00:00.000000003\",\n        \"max\": \"1970-01-01 00:00:00.000001792\",\n        \"num_unique_values\": 9,\n        \"samples\": [\n          1792.0,\n          \"\",\n          964.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"order_status\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          2,\n          \"\",\n          \"Order\\\\rFinished\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"customer\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          16,\n          \"\",\n          \"Carlos Soltero\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"order_date\",\n      \"properties\": {\n        \"dtype\": \"date\",\n        \"min\": \"1970-01-01 00:00:00.000000020\",\n        \"max\": \"2012-10-01 00:00:00\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"\",\n          \"2011-04-14 12:00:00\",\n          \"20\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"order_quantity\",\n      \"properties\": {\n        \"dtype\": \"date\",\n        \"min\": \"1970-01-01 00:00:00.000000006\",\n        \"max\": \"1970-01-01 00:00:00.000000049\",\n        \"num_unique_values\": 9,\n        \"samples\": [\n          49.0,\n          \"\",\n          26.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sales\",\n      \"properties\": {\n        \"dtype\": \"date\",\n        \"min\": \"1970-01-01 00:00:00.000000020\",\n        \"max\": \"1970-01-01 00:00:00.024056460\",\n        \"num_unique_values\": 9,\n        \"samples\": [\n          24056460.0,\n          \"\",\n          764750.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}",
              "type": "dataframe",
              "variable_name": "meta_data"
            },
            "text/html": [
              "\n",
              "  <div id=\"df-38375cb9-225e-40ff-bf85-3fd115a4052c\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>order_id</th>\n",
              "      <th>order_status</th>\n",
              "      <th>customer</th>\n",
              "      <th>order_date</th>\n",
              "      <th>order_quantity</th>\n",
              "      <th>sales</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>20.0</td>\n",
              "      <td>20</td>\n",
              "      <td>20</td>\n",
              "      <td>20</td>\n",
              "      <td>20.0</td>\n",
              "      <td>20.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>unique</th>\n",
              "      <td></td>\n",
              "      <td>2</td>\n",
              "      <td>16</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>top</th>\n",
              "      <td></td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carlos Soltero</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>freq</th>\n",
              "      <td></td>\n",
              "      <td>19</td>\n",
              "      <td>3</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>1003.65</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td>2011-05-14 12:00:00</td>\n",
              "      <td>27.3</td>\n",
              "      <td>4032125.35</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>3.0</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td>2009-11-25 00:00:00</td>\n",
              "      <td>6.0</td>\n",
              "      <td>118060.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>635.5</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td>2010-11-01 12:00:00</td>\n",
              "      <td>15.75</td>\n",
              "      <td>342045.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>964.0</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td>2011-04-14 12:00:00</td>\n",
              "      <td>26.5</td>\n",
              "      <td>764750.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>1443.75</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td>2012-02-29 06:00:00</td>\n",
              "      <td>35.75</td>\n",
              "      <td>4114145.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>1792.0</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td>2012-10-01 00:00:00</td>\n",
              "      <td>49.0</td>\n",
              "      <td>24056460.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>514.490272</td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td></td>\n",
              "      <td>13.010522</td>\n",
              "      <td>6859259.145341</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-38375cb9-225e-40ff-bf85-3fd115a4052c')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-38375cb9-225e-40ff-bf85-3fd115a4052c button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-38375cb9-225e-40ff-bf85-3fd115a4052c');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "  <div id=\"id_cfb9c84e-709f-4a05-bcda-22f083d1457d\">\n",
              "    <style>\n",
              "      .colab-df-generate {\n",
              "        background-color: #E8F0FE;\n",
              "        border: none;\n",
              "        border-radius: 50%;\n",
              "        cursor: pointer;\n",
              "        display: none;\n",
              "        fill: #1967D2;\n",
              "        height: 32px;\n",
              "        padding: 0 0 0 0;\n",
              "        width: 32px;\n",
              "      }\n",
              "\n",
              "      .colab-df-generate:hover {\n",
              "        background-color: #E2EBFA;\n",
              "        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "        fill: #174EA6;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate {\n",
              "        background-color: #3B4455;\n",
              "        fill: #D2E3FC;\n",
              "      }\n",
              "\n",
              "      [theme=dark] .colab-df-generate:hover {\n",
              "        background-color: #434B5C;\n",
              "        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "        fill: #FFFFFF;\n",
              "      }\n",
              "    </style>\n",
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('meta_data')\"\n",
              "            title=\"Generate code using this dataframe.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "    <script>\n",
              "      (() => {\n",
              "      const buttonEl =\n",
              "        document.querySelector('#id_cfb9c84e-709f-4a05-bcda-22f083d1457d button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('meta_data');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "text/plain": [
              "          order_id     order_status        customer           order_date  \\\n",
              "count         20.0               20              20                   20   \n",
              "unique                            2              16                        \n",
              "top                 Order\\rFinished  Carlos Soltero                        \n",
              "freq                             19               3                        \n",
              "mean       1003.65                                   2011-05-14 12:00:00   \n",
              "min            3.0                                   2009-11-25 00:00:00   \n",
              "25%          635.5                                   2010-11-01 12:00:00   \n",
              "50%          964.0                                   2011-04-14 12:00:00   \n",
              "75%        1443.75                                   2012-02-29 06:00:00   \n",
              "max         1792.0                                   2012-10-01 00:00:00   \n",
              "std     514.490272                                                         \n",
              "\n",
              "       order_quantity           sales  \n",
              "count            20.0            20.0  \n",
              "unique                                 \n",
              "top                                    \n",
              "freq                                   \n",
              "mean             27.3      4032125.35  \n",
              "min               6.0        118060.0  \n",
              "25%             15.75        342045.0  \n",
              "50%              26.5        764750.0  \n",
              "75%             35.75       4114145.0  \n",
              "max              49.0      24056460.0  \n",
              "std         13.010522  6859259.145341  "
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "meta_data=df.describe(include='all').fillna('')\n",
        "display(meta_data)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0w6U9yg_QIrY",
        "outputId": "7702ae4a-63ef-49b1-a97e-b694cee22d40"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "['Order ID: 3 has status Order\\\\rFinished\\nCustomer is Muhammed Mac\\\\rIntyre.Order Date is 2010-10-13 00:00:00.Quantity Ordered is 6.Total Sales amount is 523080.', 'Order ID: 293 has status Order\\\\rFinished\\nCustomer is Barry French.Order Date is 2012-10-01 00:00:00.Quantity Ordered is 49.Total Sales amount is 20246040.', 'Order ID: 483 has status Order\\\\rFinished\\nCustomer is Clay Rozendal.Order Date is 2011-07-10 00:00:00.Quantity Ordered is 30.Total Sales amount is 9931519.', 'Order ID: 515 has status Order\\\\rFinished\\nCustomer is Carlos Soltero.Order Date is 2010-08-28 00:00:00.Quantity Ordered is 19.Total Sales amount is 788540.', 'Order ID: 613 has status Order\\\\rFinished\\nCustomer is Carl Jackson.Order Date is 2011-06-17 00:00:00.Quantity Ordered is 12.Total Sales amount is 187080.', 'Order ID: 643 has status Order\\\\rFinished\\nCustomer is Monica Federle.Order Date is 2011-03-24 00:00:00.Quantity Ordered is 21.Total Sales amount is 5563640.', 'Order ID: 678 has status Order\\\\rReturned\\nCustomer is Dorothy Badders.Order Date is 2010-02-26 00:00:00.Quantity Ordered is 44.Total Sales amount is 456820.', 'Order ID: 807 has status Order\\\\rFinished\\nCustomer is Neola Schneider.Order Date is 2010-11-23 00:00:00.Quantity Ordered is 45.Total Sales amount is 393700.', 'Order ID: 868 has status Order\\\\rFinished\\nCustomer is Carlos Daly.Order Date is 2012-06-08 00:00:00.Quantity Ordered is 32.Total Sales amount is 1433680.', 'Order ID: 933 has status Order\\\\rFinished\\nCustomer is Claudia Miner.Order Date is 2012-08-04 00:00:00.Quantity Ordered is 15.Total Sales amount is 161220.', 'Order ID: 995 has status Order\\\\rFinished\\nCustomer is Neola Schneider.Order Date is 2011-05-30 00:00:00.Quantity Ordered is 46.Total Sales amount is 3630980.', 'Order ID: 998 has status Order\\\\rFinished\\nCustomer is Allen Rosenblatt.Order Date is 2009-11-25 00:00:00.Quantity Ordered is 16.Total Sales amount is 496520.', 'Order ID: 1154 has status Order\\\\rFinished\\nCustomer is Sylvia Foulston.Order Date is 2012-02-14 00:00:00.Quantity Ordered is 44.Total Sales amount is 8924460.', 'Order ID: 1344 has status Order\\\\rFinished\\nCustomer is Jim Radford.Order Date is 2012-04-15 00:00:00.Quantity Ordered is 15.Total Sales amount is 1669808.', 'Order ID: 1412 has status Order\\\\rFinished\\nCustomer is Carlos Soltero.Order Date is 2010-03-12 00:00:00.Quantity Ordered is 13.Total Sales amount is 118060.', 'Order ID: 1539 has status Order\\\\rFinished\\nCustomer is Carl Ludwig.Order Date is 2011-03-09 00:00:00.Quantity Ordered is 33.Total Sales amount is 1023660.', 'Order ID: 1540 has status Order\\\\rFinished\\nCustomer is Don Miller.Order Date is 2012-08-04 00:00:00.Quantity Ordered is 30.Total Sales amount is 161800.', 'Order ID: 1702 has status Order\\\\rFinished\\nCustomer is Annie Cyprus.Order Date is 2011-05-06 00:00:00.Quantity Ordered is 23.Total Sales amount is 134480.', 'Order ID: 1761 has status Order\\\\rFinished\\nCustomer is Carl Ludwig.Order Date is 2010-12-23 00:00:00.Quantity Ordered is 25.Total Sales amount is 24056460.', 'Order ID: 1792 has status Order\\\\rFinished\\nCustomer is Carlos Soltero.Order Date is 2010-11-08 00:00:00.Quantity Ordered is 28.Total Sales amount is 740960.']\n"
          ]
        }
      ],
      "source": [
        "documents = []\n",
        "for _, row in df.iterrows():\n",
        "  doc = (\n",
        "      f\"Order ID: {row['order_id']} has status {row['order_status']}\\n\"\n",
        "      f\"Customer is {row['customer']}.\"\n",
        "      f\"Order Date is {row['order_date']}.\"\n",
        "      f\"Quantity Ordered is {row['order_quantity']}.\"\n",
        "      f\"Total Sales amount is {row['sales']}.\"\n",
        "  )\n",
        "  documents.append(doc)\n",
        "\n",
        "print(documents)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tA5el32BMTZT",
        "outputId": "53bf1cc0-38c1-4a53-cf84-a45a422ce4b4"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 939,
          "referenced_widgets": [
            "5a3232d826ac4406b81f213b6d3556e1",
            "3f00f3b710a34d4cb6016e47ace9fc74",
            "74cde16021bd43e9a965ec8e8f0c9e58",
            "3b6140559bfc4ab0bb9cbf1df5fd14da",
            "48b3558790b04c86978f22d62c1cb4cd",
            "c294fdddfb3949b39efa7896d2ce5590",
            "952dda8d9cf64e5390fcd7f5867699f9",
            "12a9b565ee32475bb9bdf8efc94a9f33",
            "b6b7eec7293e4ad4958e56e475614bab",
            "5f6a8db9fc3f4799989ee485c60e8246",
            "56888b7706b84a309055992da07c2593",
            "6adcb1e88809401aaa06c2c7f8988b8b",
            "1d63b80b93b241bca6c43a3d0cf4d734",
            "aade5f9baf5849a1b27973d461fb838d",
            "af72004be3464c8c9ce5543627b3fc73",
            "1bef47c0412e4d7c9147a343a7639f21",
            "550432f3de6a417d9d3702901793fdff",
            "a7bd85d7a21e413b9a715203acdc53da",
            "9a9283a5d4ba482ea728e1704d79197c",
            "f1aeb0b053c145a9af53c2e1bae2e675",
            "d3b549df068347ed94e3c3f44cf44cbc",
            "7418612fac1f43fdab018aa11ef4de18",
            "983e9b6828044fad86d04fe1be19b377",
            "6fac54dcc37741d58411996e90d44a57",
            "297c3f0861214da5be4e35081c95e4ad",
            "144e8479163144fcacaa9db60f761fcb",
            "89098e6611794bf3828643310a844362",
            "850a17de386c469cac594e7a580cda8f",
            "e2f836a3db9f4a3480ca18161a1dccc0",
            "47efda45b094412281c187802ae4f09f",
            "e06bee74af9e40e397ffd02497113adb",
            "1effde4945774bf78474fa29158fc0ff",
            "4ac3d690c0294561a95ffcffd003effd",
            "9dba019402c045fd88a99a7117cb7008",
            "8c8ac1f97a1c4b77b825d2e24b9b34b0",
            "8bb53b981b7343ae9a6fa22948161359",
            "272752abd08346fa81ce98bb266305e5",
            "979d2b4208ec460398668ccfb13992e2",
            "b2c64f3dc68248e1aad02eb0015ef150",
            "4f5f67befc004f5c9f232b874baabbc8",
            "e2db7fc48b4442b98ebff2fc2e6f55ef",
            "ad03feaf58f74ca18fdd6c5923900e5e",
            "5b4bd626a11145689d797c4e0d736ffc",
            "6e3d7496c7c84854b9b6c027660821b5",
            "333680dedbd345a5a3229d8cc212a008",
            "8a3ea6fd106d4c20b90a960be0d76b1d",
            "f947b96bd7ee47b3b5a4a04e2cf21f73",
            "4baafbc9c9384ecba80e3c74728fb052",
            "d9f884ada7ca42ca9836e16f123d54c3",
            "87c7740f4e394d668ff91fde3f2b7fa7",
            "7848ed3200864c53a6b92a3a5c8ba480",
            "09dde40e422743df9e248b99e40d8e89",
            "7f65ec3ea5174341af745ece03ecbbf2",
            "a746d1c9fa0b443ba416aec93375218c",
            "cd6a20ee2305468f9418d2bde1cb3b42",
            "8be863e0b3c64eaca81775eb93f14036",
            "7fa6fb670a854f35b0302a18e12f4244",
            "6adee915c3e545789246a6a7c93bcbf4",
            "b6bea479e9ad42b29f389798015af833",
            "6598e116bb6f4c54a0838695da8c63ad",
            "2d8183792be241049baa20544b4ca4c7",
            "15eac41da7f747029bfa2c850a0d7ba0",
            "b3fb3032c8094faaaa7b996bd5c1a146",
            "378844bf6b9d415c8303c6e3cd8d4383",
            "606e4bf66b9f4d5e8cf7aff78fa8ce7c",
            "1eae6f9b3cb9436fbb744df8746fee49",
            "6dfed3d9310e40a7abb48102f96195eb",
            "1dd52fdd1d904a95b413a86b2f0fd255",
            "7581f8209181486b8c17f2dc734dd9c7",
            "3d8a5871f95344a89b31d3ca0040c1f3",
            "466c02f6d97b4467a40f5878b1a70545",
            "2410a58216c7477bbda4af51e0b50cf8",
            "a08c2faebac641baaed00676414e529e",
            "c889dca96eb046e0b5246ef40536283a",
            "12801cafb88a496a84dfaa41cf1041db",
            "71857d52e38c4378abdc206ecf660e01",
            "63843edc72a84dd0bf1d61acd896946f",
            "bdc4ddb8db054400a22b891910b69d97",
            "71640b92b0104754855864d2da4fef96",
            "7a00f7323af14576aa2581ad099a50ee",
            "920a9e086db341168452746469a24abc",
            "5356c1018a1a4ba49d3405c7f6e52a03",
            "ae3a81e9bf5d4aaf84550df196e786d4",
            "072848735d5242c686409aa43849f9a6",
            "720545c1dc164479ba9c5e025e6570df",
            "794484760eb442e9b9866bc56466120f",
            "b971bf92d7c84d4d8aec5aaf3a66f9d0",
            "f7f0e4797469462c833b96365a89cb46",
            "793e25f6032142ab9b1e7577ce63a0f1",
            "6056b33c9479490f9f9e93e8f07158f2",
            "2276e0d4c1af45a884aa317291208920",
            "43edd8fb79c0483d9ba63b2569be1da7",
            "8d28e85ee7994104abec8f017be1324e",
            "71fc73acd236477f91636dae4dcc7817",
            "a05a15f1abff49c9bd0a3a4ab796a2ee",
            "f24ea98e557945d5b5af0085aefaf2e6",
            "0cee572b658246c0ad5cc0ba55115ccb",
            "7f0125fe09da42d5a35e043edd8dca11",
            "a30b112a407c462cbc7f1d56cb4fca75",
            "2b77a1e97fd14d1580bfbf45705457b5",
            "90690271655e423fac84f98cb145002b",
            "054de63069264aaa9c02f7daf656d022",
            "3ac1751e3deb46b6ab44f3542bb9484e",
            "f0271fac27c244edbbe1cc1e28b90f07",
            "6b412eb741e346849b9a394bf637867e",
            "8398ab9617214ee28a801a8b0f77d16c",
            "3c68fe860a3240f2933fad46c55e4ace",
            "e205f3c9ec8944a4a1706b18fbe8a5ff",
            "695b6d8f2fbf4053b8ea5b15cc55e473",
            "da1e28258d2847409cc184f6fd01bba9",
            "dd5a0fb8a1ea4317ac90b7b196d85bbe",
            "69968d3a78ca4fe997b251d8158d4145",
            "a20e4f5145b64dfa8f4b5ce4706c68cc",
            "7fa0ee5f62f64bdba7aa3a7b8a60f6bc",
            "e46184f7d15c4166ba8add3a55a08ca4",
            "c2da60f3ace44f58b354b353064dbd18",
            "726c7afbf243431fbb23eb6b1d952796",
            "6bdf155a57ec4ab1a1c3ffac62226a5e",
            "b88a36748af74d26a0a3b5e9cc265af9",
            "fbf82d86d8314722b9e7061e051d7ac0",
            "572ebe16f1f54c24bb53595afe11b34f",
            "86512b91b4d643faae0b99390813c598",
            "d66a785eae2449c3b748ed923b278a38",
            "9761dace89e245a4a4e467e4af64e8ad",
            "9024e770bec1405d97b1c7b637967751",
            "792154abdfb443c8a30a73068be23d13",
            "409878d01b97419586eb53b813b8e8d4",
            "7bea44cd7e074d6fb175c1efce63fc69",
            "77ececd1f05846b6a7f599fecce45781",
            "bb783e7146864ca2aa602a3e01d0d177",
            "1f4a71cf432d429c98e22c1ec78a4d6c",
            "53600991d1504b31a6942ce2e8082ce8",
            "b4c7e37ca078478086b167f95105f5a1",
            "404f063992e24636a9bc1627b42d7872",
            "937b3c3534ea4a92bd40386f9d5df178",
            "09710413ebcd44e0a579bf6ce6f8b854",
            "b110261a1da645dfa03305048d2a8197",
            "3d11a2dde31d4ec9af59459603de867e",
            "a0450aa028c94819bf5c6beb2316c7f8",
            "a7122ac6c8e64d4aa011117d04f47397",
            "5ae045f5466344268ee8abc4eec674bd",
            "78d03f2be5124881b56a2bb7df7522ff",
            "8dd40c90a4664531a4d83f0215fb5016"
          ]
        },
        "id": "9p1UO7-hKCKS",
        "outputId": "a4282262-057d-4102-f537-83e4f81e2637"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: \n",
            "The secret `HF_TOKEN` does not exist in your Colab secrets.\n",
            "To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.\n",
            "You will be able to reuse this secret in all of your notebooks.\n",
            "Please note that authentication is recommended but still optional to access public models or datasets.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "5a3232d826ac4406b81f213b6d3556e1",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "6adcb1e88809401aaa06c2c7f8988b8b",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "983e9b6828044fad86d04fe1be19b377",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "README.md: 0.00B [00:00, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "9dba019402c045fd88a99a7117cb7008",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "333680dedbd345a5a3229d8cc212a008",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n",
            "WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n"
          ]
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "8be863e0b3c64eaca81775eb93f14036",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "6dfed3d9310e40a7abb48102f96195eb",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "BertModel LOAD REPORT from: sentence-transformers/all-miniLM-L6-v2\n",
            "Key                     | Status     |  | \n",
            "------------------------+------------+--+-\n",
            "embeddings.position_ids | UNEXPECTED |  | \n",
            "\n",
            "Notes:\n",
            "- UNEXPECTED\t:can be ignored when loading from different task/architecture; not ok if you expect identical arch.\n"
          ]
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "bdc4ddb8db054400a22b891910b69d97",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "tokenizer_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "793e25f6032142ab9b1e7577ce63a0f1",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "vocab.txt: 0.00B [00:00, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "2b77a1e97fd14d1580bfbf45705457b5",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "tokenizer.json: 0.00B [00:00, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "dd5a0fb8a1ea4317ac90b7b196d85bbe",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "86512b91b4d643faae0b99390813c598",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "b4c7e37ca078478086b167f95105f5a1",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[[-0.0531115  -0.0253893   0.0304186  ... -0.03634322 -0.046556\n",
            "   0.03768994]\n",
            " [-0.04064846 -0.00404917 -0.01074561 ... -0.02884728 -0.06390351\n",
            "  -0.0145968 ]\n",
            " [-0.0597324  -0.01271612 -0.00633057 ... -0.00966161 -0.07273875\n",
            "  -0.01768629]\n",
            " ...\n",
            " [ 0.00922527  0.0265976   0.05770034 ... -0.00990441 -0.01055641\n",
            "  -0.02974943]\n",
            " [-0.08787361  0.03158558  0.02962808 ... -0.02135037 -0.01379733\n",
            "   0.00251404]\n",
            " [-0.06940062  0.01317554 -0.00728172 ...  0.01527077 -0.08632758\n",
            "   0.00168001]]\n"
          ]
        }
      ],
      "source": [
        "#embeddings\n",
        "model=SentenceTransformer('sentence-transformers/all-miniLM-L6-v2')\n",
        "embeddings=model.encode(documents,convert_to_numpy=True,show_progress_bar=True)\n",
        "print(embeddings)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nM22evg8KCOI"
      },
      "outputs": [],
      "source": [
        "dimension=embeddings.shape[1]\n",
        "index=faiss.IndexFlatL2(dimension)\n",
        "index.add(embeddings)\n",
        "faiss.write_index(index,'faiss_index.faiss')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kplCSYjKKCRJ"
      },
      "outputs": [],
      "source": [
        "def retrieve_context(query,k=3):\n",
        "  query_embeddings=model.encode([query])\n",
        "  distance,indices=index.search(query_embeddings,k)\n",
        "  return '\\n'.join([documents[i] for i in indices[0]])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QX34KsM8KCUR"
      },
      "outputs": [],
      "source": [
        "generation_config={'temperature':0.4,'max_output_tokens':512}"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "19XtoS_qKCYq"
      },
      "outputs": [],
      "source": [
        "gemini_model=genai.GenerativeModel(model_name='models/gemini-2.5-flash',generation_config=generation_config)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jps4VX3kKCcG"
      },
      "outputs": [],
      "source": [
        "chat_history=[]\n",
        "def chat_with_bot(user_input):\n",
        "  global chat_history\n",
        "  context= retrieve_context(user_input)\n",
        "  prompt=f\"\"\" you are a conversational data analyst chatbot. please answer the user question using the given context below.\n",
        "              context={context}\n",
        "              user_question={user_input}\n",
        "              Rules:\n",
        "              1.Be conversational\n",
        "              2.Only answer using the given context\n",
        "              3.Say don't have enough imformation, if you don't know the answer.\"\"\"\n",
        "  response=gemini_model.generate_content(prompt)\n",
        "  answer=response.text\n",
        "  chat_history.append({'user':user_input,'bot':answer})\n",
        "  return answer\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 225
        },
        "id": "1PWM0X073LuM",
        "outputId": "1f22b608-e833-4b3a-9a4f-bf9383df647e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Conversational chatbot is ready for sales document\n",
            "type exit to stop the conversation \n",
            "\n",
            "User:status of order 293\n",
            "Bot:Hello there!\n",
            "\n",
            "Looking at the data, Order ID 293 has a status of **Order Finished**.\n",
            "\n",
            "Is there anything else I can help you with regarding this order or others?\n",
            "____________________________________________________________\n",
            "User:exit\n",
            "Good Bye!\n"
          ]
        }
      ],
      "source": [
        "print('Conversational chatbot is ready for sales document')\n",
        "print(\"type exit to stop the conversation \\n\")\n",
        "while True:\n",
        "  user_input=input(\"User:\")\n",
        "  if user_input.lower() in ['exit','stop','bye']:\n",
        "    print(\"Good Bye!\")\n",
        "    break\n",
        "  response=chat_with_bot(user_input)\n",
        "  print(f'Bot:{response}')\n",
        "  print('_'*60)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rIo9Qybu3Lw1"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "x0jeOdta3Lzn"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ODrVtEqI3L28"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }

  },
  "nbformat": 4,
  "nbformat_minor": 0
}
