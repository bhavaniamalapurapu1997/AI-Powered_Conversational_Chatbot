{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
  },
	"cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9lhna4pH0c0K"
      },
      "outputs": [],
      "source": [
        "# Step -1 Download the data from the link shared in the Chat\n",
        "# Step -2 Upload the data in Google Colab File Section\n",
        "# Step -3 Copy the path of the file, create a variable path and paste"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## Step -4 Installation of Libraries\n",
        "#### Libraries to get installed and also to put the names in requirements.txt\n",
        "!pip install tabula-py pandas sentence-transformers faiss-cpu google-generativeai"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "ZOV4lS5h8oxm",
        "outputId": "aeb488d0-9a8c-426f-fe98-ce745e7bec50"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting tabula-py\n",
            "  Downloading tabula_py-2.10.0-py3-none-any.whl.metadata (7.6 kB)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)\n",
            "Requirement already satisfied: sentence-transformers in /usr/local/lib/python3.12/dist-packages (5.2.0)\n",
            "Collecting faiss-cpu\n",
            "  Downloading faiss_cpu-1.13.2-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (7.6 kB)\n",
            "Requirement already satisfied: google-generativeai in /usr/local/lib/python3.12/dist-packages (0.8.6)\n",
            "Requirement already satisfied: numpy>1.24.4 in /usr/local/lib/python3.12/dist-packages (from tabula-py) (2.0.2)\n",
            "Requirement already satisfied: distro in /usr/local/lib/python3.12/dist-packages (from tabula-py) (1.9.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.3)\n",
            "Requirement already satisfied: transformers<6.0.0,>=4.41.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.57.6)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.67.1)\n",
            "Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (2.9.0+cpu)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.6.1)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.16.3)\n",
            "Requirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (0.36.0)\n",
            "Requirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.15.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from faiss-cpu) (25.0)\n",
            "Requirement already satisfied: google-ai-generativelanguage==0.6.15 in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (0.6.15)\n",
            "Requirement already satisfied: google-api-core in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.29.0)\n",
            "Requirement already satisfied: google-api-python-client in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.188.0)\n",
            "Requirement already satisfied: google-auth>=2.15.0 in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.43.0)\n",
            "Requirement already satisfied: protobuf in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (5.29.5)\n",
            "Requirement already satisfied: pydantic in /usr/local/lib/python3.12/dist-packages (from google-generativeai) (2.12.3)\n",
            "Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in /usr/local/lib/python3.12/dist-packages (from google-ai-generativelanguage==0.6.15->google-generativeai) (1.27.0)\n",
            "Requirement already satisfied: googleapis-common-protos<2.0.0,>=1.56.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core->google-generativeai) (1.72.0)\n",
            "Requirement already satisfied: requests<3.0.0,>=2.18.0 in /usr/local/lib/python3.12/dist-packages (from google-api-core->google-generativeai) (2.32.4)\n",
            "Requirement already satisfied: cachetools<7.0,>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (6.2.4)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (0.4.2)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.12/dist-packages (from google-auth>=2.15.0->google-generativeai) (4.9.1)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.20.3)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2025.3.0)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.3)\n",
            "Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.2.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.6.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (2025.11.3)\n",
            "Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.22.2)\n",
            "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers<6.0.0,>=4.41.0->sentence-transformers) (0.7.0)\n",
            "Requirement already satisfied: httplib2<1.0.0,>=0.19.0 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (0.31.1)\n",
            "Requirement already satisfied: google-auth-httplib2<1.0.0,>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (0.3.0)\n",
            "Requirement already satisfied: uritemplate<5,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from google-api-python-client->google-generativeai) (4.2.0)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic->google-generativeai) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic->google-generativeai) (2.41.4)\n",
            "Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from pydantic->google-generativeai) (0.4.2)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (1.5.3)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (3.6.0)\n",
            "Requirement already satisfied: grpcio<2.0.0,>=1.33.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai) (1.76.0)\n",
            "Requirement already satisfied: grpcio-status<2.0.0,>=1.33.2 in /usr/local/lib/python3.12/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai) (1.71.2)\n",
            "Requirement already satisfied: pyparsing<4,>=3.0.4 in /usr/local/lib/python3.12/dist-packages (from httplib2<1.0.0,>=0.19.0->google-api-python-client->google-generativeai) (3.3.1)\n",
            "Requirement already satisfied: pyasn1<0.7.0,>=0.6.1 in /usr/local/lib/python3.12/dist-packages (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai) (0.6.2)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3.0.0,>=2.18.0->google-api-core->google-generativeai) (2026.1.4)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.3)\n",
            "Downloading tabula_py-2.10.0-py3-none-any.whl (12.0 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m12.0/12.0 MB\u001b[0m \u001b[31m110.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading faiss_cpu-1.13.2-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (23.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m23.8/23.8 MB\u001b[0m \u001b[31m90.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: faiss-cpu, tabula-py\n",
            "Successfully installed faiss-cpu-1.13.2 tabula-py-2.10.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Step 5 Create an Google GEMINI API Key -- Copy That API key\n",
        "### Step 6 Click on the Secrets in Google Colab and Paste the API key value and give a name and also activate the toggle"
      ],
      "metadata": {
        "id": "lOIURMwr9Mst"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "### Step -7 Import of packages\n",
        "import tabula\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import faiss\n",
        "from sentence_transformers import SentenceTransformer\n",
        "import google.generativeai as genai\n",
        "from google.colab import userdata"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "nX0m2_Eh_yyu",
        "outputId": "6e55b7b8-cfa6-401c-dc7f-3fdbb17b78e7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:torchao.kernel.intmm:Warning: Detected no triton, on systems without Triton certain kernels will not work\n",
            "/usr/local/lib/python3.12/dist-packages/google/colab/_import_hooks/_hook_injector.py:55: FutureWarning: \n",
            "\n",
            "All support for the `google.generativeai` package has ended. It will no longer be receiving \n",
            "updates or bug fixes. Please switch to the `google.genai` package as soon as possible.\n",
            "See README for more details:\n",
            "\n",
            "https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md\n",
            "\n",
            "  loader.exec_module(module)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## Step -8\n",
        "##### Load the GEMINI API KEY from Colab Secrets\n",
        "GEMINI_API_KEY = userdata.get(\"GEMINI_API_KEY\")\n",
        "if GEMINI_API_KEY is None:\n",
        "  raise ValueError(\"No GEMINI_API_KEY found in Colab Secrets\")\n",
        "### Configuring the API Key to be used\n",
        "genai.configure(api_key=GEMINI_API_KEY)"
      ],
      "metadata": {
        "id": "nBIp3gEwAO2R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "## Step 9\n",
        "### Extracting the tables from the Pdf\n",
        "pdf_path = '/content/order.pdf'\n",
        "tables = tabula.read_pdf(pdf_path, pages='all')\n",
        "df = pd.concat(tables, ignore_index=True)\n",
        "df"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 676
        },
        "collapsed": true,
        "id": "XjAkT1sABWSJ",
        "outputId": "29a26b11-f88a-4b89-db4f-c4d1d980639c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "    order_id     order_status              customer  order_date  \\\n",
              "0          3  Order\\rFinished  Muhammed Mac\\rIntyre  13/10/2010   \n",
              "1        293  Order\\rFinished          Barry French   1/10/2012   \n",
              "2        483  Order\\rFinished         Clay Rozendal   10/7/2011   \n",
              "3        515  Order\\rFinished        Carlos Soltero   28/8/2010   \n",
              "4        613  Order\\rFinished          Carl Jackson   17/6/2011   \n",
              "5        643  Order\\rFinished        Monica Federle   24/3/2011   \n",
              "6        678  Order\\rReturned       Dorothy Badders   26/2/2010   \n",
              "7        807  Order\\rFinished       Neola Schneider  23/11/2010   \n",
              "8        868  Order\\rFinished           Carlos Daly    8/6/2012   \n",
              "9        933  Order\\rFinished         Claudia Miner    4/8/2012   \n",
              "10       995  Order\\rFinished       Neola Schneider   30/5/2011   \n",
              "11       998  Order\\rFinished      Allen Rosenblatt  25/11/2009   \n",
              "12      1154  Order\\rFinished       Sylvia Foulston   14/2/2012   \n",
              "13      1344  Order\\rFinished           Jim Radford   15/4/2012   \n",
              "14      1412  Order\\rFinished        Carlos Soltero   12/3/2010   \n",
              "15      1539  Order\\rFinished           Carl Ludwig    9/3/2011   \n",
              "16      1540  Order\\rFinished            Don Miller    4/8/2012   \n",
              "17      1702  Order\\rFinished          Annie Cyprus    6/5/2011   \n",
              "18      1761  Order\\rFinished           Carl Ludwig  23/12/2010   \n",
              "19      1792  Order\\rFinished        Carlos Soltero   8/11/2010   \n",
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
              "19              28    740960  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-abbf2577-71b7-4aeb-b72f-5a563fa44bdb\" class=\"colab-df-container\">\n",
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
              "      <th>0</th>\n",
              "      <td>3</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Muhammed Mac\\rIntyre</td>\n",
              "      <td>13/10/2010</td>\n",
              "      <td>6</td>\n",
              "      <td>523080</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>293</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Barry French</td>\n",
              "      <td>1/10/2012</td>\n",
              "      <td>49</td>\n",
              "      <td>20246040</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>483</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Clay Rozendal</td>\n",
              "      <td>10/7/2011</td>\n",
              "      <td>30</td>\n",
              "      <td>9931519</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>515</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carlos Soltero</td>\n",
              "      <td>28/8/2010</td>\n",
              "      <td>19</td>\n",
              "      <td>788540</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>613</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carl Jackson</td>\n",
              "      <td>17/6/2011</td>\n",
              "      <td>12</td>\n",
              "      <td>187080</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>643</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Monica Federle</td>\n",
              "      <td>24/3/2011</td>\n",
              "      <td>21</td>\n",
              "      <td>5563640</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>678</td>\n",
              "      <td>Order\\rReturned</td>\n",
              "      <td>Dorothy Badders</td>\n",
              "      <td>26/2/2010</td>\n",
              "      <td>44</td>\n",
              "      <td>456820</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>807</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Neola Schneider</td>\n",
              "      <td>23/11/2010</td>\n",
              "      <td>45</td>\n",
              "      <td>393700</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>868</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carlos Daly</td>\n",
              "      <td>8/6/2012</td>\n",
              "      <td>32</td>\n",
              "      <td>1433680</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9</th>\n",
              "      <td>933</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Claudia Miner</td>\n",
              "      <td>4/8/2012</td>\n",
              "      <td>15</td>\n",
              "      <td>161220</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>10</th>\n",
              "      <td>995</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Neola Schneider</td>\n",
              "      <td>30/5/2011</td>\n",
              "      <td>46</td>\n",
              "      <td>3630980</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>11</th>\n",
              "      <td>998</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Allen Rosenblatt</td>\n",
              "      <td>25/11/2009</td>\n",
              "      <td>16</td>\n",
              "      <td>496520</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>12</th>\n",
              "      <td>1154</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Sylvia Foulston</td>\n",
              "      <td>14/2/2012</td>\n",
              "      <td>44</td>\n",
              "      <td>8924460</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>13</th>\n",
              "      <td>1344</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Jim Radford</td>\n",
              "      <td>15/4/2012</td>\n",
              "      <td>15</td>\n",
              "      <td>1669808</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>14</th>\n",
              "      <td>1412</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carlos Soltero</td>\n",
              "      <td>12/3/2010</td>\n",
              "      <td>13</td>\n",
              "      <td>118060</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>15</th>\n",
              "      <td>1539</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carl Ludwig</td>\n",
              "      <td>9/3/2011</td>\n",
              "      <td>33</td>\n",
              "      <td>1023660</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>16</th>\n",
              "      <td>1540</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Don Miller</td>\n",
              "      <td>4/8/2012</td>\n",
              "      <td>30</td>\n",
              "      <td>161800</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17</th>\n",
              "      <td>1702</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Annie Cyprus</td>\n",
              "      <td>6/5/2011</td>\n",
              "      <td>23</td>\n",
              "      <td>134480</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>1761</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carl Ludwig</td>\n",
              "      <td>23/12/2010</td>\n",
              "      <td>25</td>\n",
              "      <td>24056460</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19</th>\n",
              "      <td>1792</td>\n",
              "      <td>Order\\rFinished</td>\n",
              "      <td>Carlos Soltero</td>\n",
              "      <td>8/11/2010</td>\n",
              "      <td>28</td>\n",
              "      <td>740960</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-abbf2577-71b7-4aeb-b72f-5a563fa44bdb')\"\n",
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
              "        document.querySelector('#df-abbf2577-71b7-4aeb-b72f-5a563fa44bdb button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-abbf2577-71b7-4aeb-b72f-5a563fa44bdb');\n",
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
              "  <div id=\"id_9b59655c-e788-431c-9085-5366a9763e07\">\n",
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
              "    <button class=\"colab-df-generate\" onclick=\"generateWithVariable('df')\"\n",
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
              "        document.querySelector('#id_9b59655c-e788-431c-9085-5366a9763e07 button.colab-df-generate');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      buttonEl.onclick = () => {\n",
              "        google.colab.notebook.generateWithVariable('df');\n",
              "      }\n",
              "      })();\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df",
              "summary": "{\n  \"name\": \"df\",\n  \"rows\": 20,\n  \"fields\": [\n    {\n      \"column\": \"order_id\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 514,\n        \"min\": 3,\n        \"max\": 1792,\n        \"num_unique_values\": 20,\n        \"samples\": [\n          3,\n          1702,\n          1539\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"order_status\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          \"Order\\rReturned\",\n          \"Order\\rFinished\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"customer\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 16,\n        \"samples\": [\n          \"Muhammed Mac\\rIntyre\",\n          \"Barry French\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"order_date\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"num_unique_values\": 19,\n        \"samples\": [\n          \"13/10/2010\",\n          \"24/3/2011\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"order_quantity\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 13,\n        \"min\": 6,\n        \"max\": 49,\n        \"num_unique_values\": 17,\n        \"samples\": [\n          6,\n          49\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"sales\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 6859259,\n        \"min\": 118060,\n        \"max\": 24056460,\n        \"num_unique_values\": 20,\n        \"samples\": [\n          523080,\n          134480\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Step 10\n",
        "### Converting the dataframe rows to text\n",
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
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a7LUbKfJCsPE",
        "outputId": "c1746df9-4c0d-453d-da34-99af3d114372"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['Order ID: 3 has status Order\\rFinished\\nCustomer is Muhammed Mac\\rIntyre.Order Date is 13/10/2010.Quantity Ordered is 6.Total Sales amount is 523080.', 'Order ID: 293 has status Order\\rFinished\\nCustomer is Barry French.Order Date is 1/10/2012.Quantity Ordered is 49.Total Sales amount is 20246040.', 'Order ID: 483 has status Order\\rFinished\\nCustomer is Clay Rozendal.Order Date is 10/7/2011.Quantity Ordered is 30.Total Sales amount is 9931519.', 'Order ID: 515 has status Order\\rFinished\\nCustomer is Carlos Soltero.Order Date is 28/8/2010.Quantity Ordered is 19.Total Sales amount is 788540.', 'Order ID: 613 has status Order\\rFinished\\nCustomer is Carl Jackson.Order Date is 17/6/2011.Quantity Ordered is 12.Total Sales amount is 187080.', 'Order ID: 643 has status Order\\rFinished\\nCustomer is Monica Federle.Order Date is 24/3/2011.Quantity Ordered is 21.Total Sales amount is 5563640.', 'Order ID: 678 has status Order\\rReturned\\nCustomer is Dorothy Badders.Order Date is 26/2/2010.Quantity Ordered is 44.Total Sales amount is 456820.', 'Order ID: 807 has status Order\\rFinished\\nCustomer is Neola Schneider.Order Date is 23/11/2010.Quantity Ordered is 45.Total Sales amount is 393700.', 'Order ID: 868 has status Order\\rFinished\\nCustomer is Carlos Daly.Order Date is 8/6/2012.Quantity Ordered is 32.Total Sales amount is 1433680.', 'Order ID: 933 has status Order\\rFinished\\nCustomer is Claudia Miner.Order Date is 4/8/2012.Quantity Ordered is 15.Total Sales amount is 161220.', 'Order ID: 995 has status Order\\rFinished\\nCustomer is Neola Schneider.Order Date is 30/5/2011.Quantity Ordered is 46.Total Sales amount is 3630980.', 'Order ID: 998 has status Order\\rFinished\\nCustomer is Allen Rosenblatt.Order Date is 25/11/2009.Quantity Ordered is 16.Total Sales amount is 496520.', 'Order ID: 1154 has status Order\\rFinished\\nCustomer is Sylvia Foulston.Order Date is 14/2/2012.Quantity Ordered is 44.Total Sales amount is 8924460.', 'Order ID: 1344 has status Order\\rFinished\\nCustomer is Jim Radford.Order Date is 15/4/2012.Quantity Ordered is 15.Total Sales amount is 1669808.', 'Order ID: 1412 has status Order\\rFinished\\nCustomer is Carlos Soltero.Order Date is 12/3/2010.Quantity Ordered is 13.Total Sales amount is 118060.', 'Order ID: 1539 has status Order\\rFinished\\nCustomer is Carl Ludwig.Order Date is 9/3/2011.Quantity Ordered is 33.Total Sales amount is 1023660.', 'Order ID: 1540 has status Order\\rFinished\\nCustomer is Don Miller.Order Date is 4/8/2012.Quantity Ordered is 30.Total Sales amount is 161800.', 'Order ID: 1702 has status Order\\rFinished\\nCustomer is Annie Cyprus.Order Date is 6/5/2011.Quantity Ordered is 23.Total Sales amount is 134480.', 'Order ID: 1761 has status Order\\rFinished\\nCustomer is Carl Ludwig.Order Date is 23/12/2010.Quantity Ordered is 25.Total Sales amount is 24056460.', 'Order ID: 1792 has status Order\\rFinished\\nCustomer is Carlos Soltero.Order Date is 8/11/2010.Quantity Ordered is 28.Total Sales amount is 740960.']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Create Embeddings - Step 11\n",
        "embedding_model = SentenceTransformer('all-MiniLM-L6-v2')\n",
        "embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)\n",
        "print(embeddings)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 525,
          "referenced_widgets": [
            "99c13f8cb44a4310a6ecbba63750ad7b",
            "b84cd8060c1e42a7bc82d705574897e6",
            "ac053be9cab247768e20675898c83fa0",
            "102c7dc303b34bb89d280497750f0661",
            "dc9b33baa3b3485084aa3b156edd9237",
            "53905349652b459d9cba253d33700b08",
            "0c6a59c788a94aa2a4edced6badea2c7",
            "f5c39647752f4fa4a901d4975cd1bf8a",
            "85cff64de7da48389d1aac70bef0c02d",
            "0ed51d8c6b7a4b2cb819c430a1231e49",
            "bd329fde73204e97a254278c0c55e91d",
            "a6ac02a67ca84ff587992eafeb189102",
            "9612e475367d4e6784b8ad7ec0b45f51",
            "62f78f32848a470386abe978f9528d51",
            "52753968abf94990b6bb05e3dc8b041f",
            "108eac13661d4bc2bb8ce8fd4fd1a88a",
            "cc4b35cd94b24ece92c0257f7a10104e",
            "1445fbb13b904fac94655fdd1fbc3ed7",
            "dd8c8666c150452b94b72c778029d596",
            "6af02add0e884e8280334812ceb77332",
            "ec7588d9133549fd83cd803a341aae17",
            "2bcac8573d644a47b3d6c89017e7f004",
            "a7dc957c33cc462d9b8df45106091c5a",
            "eeda950cf5d54bca87f50e7fb8ee6614",
            "b2e10dd773d142628a59f6783a242c65",
            "74f62c6b6b9346e5bf9905e4d00e6b70",
            "c7ee697f9b774e39b4bdef67754e3fad",
            "be0d48ca5dff47dbbbf9fa7727767543",
            "7aef49a49ea448178f5b57f14ccd1bdc",
            "169b07f47fb6419f9359b8eacbd84099",
            "faed92238256456b964ae81387083a1a",
            "770a2e20e47543febb8a74927c705675",
            "e6220af430884ceab68ab0d781e53a95",
            "1bc3b75adc8e4f64ab6dfff1308b3626",
            "5f4febc73aee4de49aadb1f6f16ef2c1",
            "9fbd4729146548f98db609de913c2a9b",
            "4db72f17acc642e3b6d205e4ddeb7fc2",
            "ff76259ae040423cb612181e69b1556d",
            "23b1500e46724fa3979c14f9c79c7aeb",
            "6f60216ad29d4bb7b4715320e649a5ac",
            "50f896b9ea7d4e23b3cd608b0e6ae2a9",
            "851315b8544e407f8ce729553f5741f1",
            "cb32f90671f54c6e95560f69d61b0bd4",
            "b8d6818ed36b416c9ee1dff9a8175869",
            "985b7f8efb744902aee5e1b69b21a725",
            "2423c7a27d364fd5a6488ea08a21da46",
            "eece8733fec6417385df9b8ceade71f7",
            "52bb6f7d1e1c4d1e966f696dc8d12e78",
            "0781f772e9f24c7f9b689f074b270ffd",
            "9946b998849a4c8da822b70d6cd8b8ae",
            "4caecda763e84938838087c09beb77d6",
            "da08454e76824d3c8cc975feaaf5de9c",
            "32a791a479b74b35948810bbdc5b54e9",
            "d2f904b552ca477d8d9d95a4ad69c3ff",
            "4b39eb84c0604d02be7acc665497f650",
            "2e7ddef5e75b4df6b7819db448e123ca",
            "34555c0742404937a0b39b0fd2e09cb5",
            "17eaf75dcffd42fbbc43f32003795b51",
            "d498d178aca84a97a33c1bc3023629d0",
            "15dff2df6a80431097fe9a2d089fd786",
            "e1ce7ed346274724b7a00e8ce4c89c6b",
            "5a02ea10011a455cad3d071f2968b55b",
            "d775aa232225461fbf171599efa09182",
            "8230ce711ee34a8e92ae6894184bc2ae",
            "8ef5fa0ac3ed4bb98ccc89bb2e59d3ff",
            "783adacf415d4dd2b9451ff0538aac9f",
            "6e270387f2004f94ac4a4682cfdbbe2b",
            "121256acaa424f9eac9080c011683161",
            "4da68d745e104a15bfe351a917663b8c",
            "f422e247fb1d47c3bd91bce1dfb508ef",
            "101a05b0572e49d4a69e5cc769c8f154",
            "4bc7f1b6744245cda279c4a9c4cd3a4b",
            "bbd5af291e4649ce8254b40acdcb63a6",
            "bac59a355dc841efa765616d2f7e655c",
            "9ee95155e5cf4eb7895da8c17bba73af",
            "ce55151b6dfe41b092c4e1c014a7af81",
            "572aab76fdf349a282e996506cc22b31",
            "61b1c1d7f8af4bca95eeb4b9588b6d14",
            "1715d0d8318e4a68841dd9e7a3ee5b86",
            "1b45d214d2244fa9b0b47137a86e450a",
            "21e3476b31ec4d3e85211ebcb51a0fe7",
            "bf9cbd62ca7c4e52903a9e888d95b260",
            "f701847d26cc4bc998fef9bcfdb0086d",
            "ab62bf623bcd4721b8d210a03b5feb23",
            "be7bb729d1944e6fa2293aa7ad112108",
            "db5c5b86ea954fc1b97ab11f4d115d56",
            "38bde8000afa4f248489ecfbe4d2aa41",
            "56adf70c5463451e9d368550147d6f7c",
            "a0fab23afb304fe8a78b1e248c31395c",
            "36ef2794294946a08e8fd356f5a69968",
            "44de3418f6944a6d98430405d6b76a41",
            "54141a8ef445434aa001e3d92311baa3",
            "52f9fdd56aad41158dfae7fc44ebd27f",
            "487e0aa3e1e74244b3515ad41c50e7a6",
            "04802a7c56f542868d6c61fc01f35a4d",
            "7afde1c800dc4707a2e0493d6e430cc1",
            "0807b29d3c184ed9b0b3056700d93fff",
            "56c05832588b4699bb16836de2349920",
            "bac605b18b994f75a0dd419980866983",
            "85277b1ba5444ff4b7610de77b159c23",
            "9f47fa295f8841e7b7147332755fff9f",
            "bbef0b717bc7415c9f6166f0f6c64b5a",
            "675017af4c6f441794a3014cc02cfc1b",
            "f754c53654d44f1793f2ebbe9c39e3e4",
            "12b8e33585994e9793349298a6ea82ec",
            "0dc6c5b5ce054b06b13b026f7425b9f2",
            "d0345ba5c7814ce7ac484ca7bf166f2c",
            "d7f2a55d4a6c40c3bcd99ef4507c2bf6",
            "cc1a6064551947caac93194188258c9f",
            "52d31d2650af48ea99d6ac3ebd13b930",
            "e73fda00a28747d28397236c76b07d54",
            "dc6005de1d484c68a55a705ea03ca0b5",
            "67d5b54cb54b46469a6b4afd265fcaed",
            "91638a381e164d5c95bdfe0b4cf02e3f",
            "d07be3cee3ad44a6b445651eb32f6142",
            "cf93423e14d94c28a8a4c45f237c08d0",
            "1fb302e07ce64ed7a49641a3f6e16153",
            "d4104da41f6f4e0ab2404b16b3f5b65b",
            "25d26bc6940a48a6b176e4e47902c5b0",
            "715984157400459b8238e068a5867d6e",
            "f97ccaa9bf704bea8600c9a860280595",
            "2e45eca7938047858367ea5f4b8aa300",
            "146e121f9e6f4146911fed3c78cea19f",
            "1dc51e90735b485c9b77e6ccf3993ff6",
            "726e014d892a42aba44465d29c7c2115",
            "c84445cdada4458a9174ecd9cc63e072",
            "8cdd27de5d7d4b14a7c4479dc1ebaac0",
            "ce4d8dea859f45d1b0dc61f96e11eb5a",
            "9d52c16052314799b55dd10397f1d504",
            "cf3d433f116c49efaf1bed85dcac970a",
            "c4465d43d68f4a6dbe242cdf5e989c6f",
            "24daac8cc4d24f63aed28be8b3ccf87f"
          ]
        },
        "collapsed": true,
        "id": "m0OiJcw6FMB6",
        "outputId": "0e9f11ad-3259-408c-c644-f6169281c179"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
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
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "99c13f8cb44a4310a6ecbba63750ad7b"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "a6ac02a67ca84ff587992eafeb189102"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "README.md: 0.00B [00:00, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "a7dc957c33cc462d9b8df45106091c5a"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "1bc3b75adc8e4f64ab6dfff1308b3626"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "985b7f8efb744902aee5e1b69b21a725"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "2e7ddef5e75b4df6b7819db448e123ca"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "tokenizer_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "6e270387f2004f94ac4a4682cfdbbe2b"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "vocab.txt: 0.00B [00:00, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "61b1c1d7f8af4bca95eeb4b9588b6d14"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "tokenizer.json: 0.00B [00:00, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "a0fab23afb304fe8a78b1e248c31395c"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "85277b1ba5444ff4b7610de77b159c23"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "e73fda00a28747d28397236c76b07d54"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Batches:   0%|          | 0/1 [00:00<?, ?it/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "2e45eca7938047858367ea5f4b8aa300"
            }
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Store the embeddings in FAISS  -Step -12\n",
        "dimension = embeddings.shape[1]\n",
        "index = faiss.IndexFlatL2(dimension)\n",
        "index.add(embeddings)\n",
        "faiss.write_index(index, 'faiss_index.faiss')  #### I am saving the index file as well"
      ],
      "metadata": {
        "collapsed": true,
        "id": "VVmxw6NUHeQk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "### Retrieval Function ### Step 13\n",
        "def retrive_context(query, k=3):\n",
        "  query_embedding = embedding_model.encode([query])\n",
        "  distances, indices = index.search(query_embedding, k)\n",
        "  return \"\\n\".join([documents[i] for i in  indices[0]])"
      ],
      "metadata": {
        "id": "wOkbANQ2JA-_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "### GEMINI Model Configuration ### Step -14\n",
        "### max_output_tokens , temp (helps the model to create response  - creative or deterministic [0-1])\n",
        "## The higher the temp - the higher the creativity is , lower the temp  - the more deterministic respose.\n",
        "generation_config = {\n",
        "    \"temperature\": 0.4,\n",
        "    \"max_output_tokens\": 512\n",
        "}\n",
        "print(generation_config)\n",
        "gemini_model = genai.GenerativeModel(model_name='models/gemini-2.5-flash',generation_config=generation_config)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OduN4LfqHl7A",
        "outputId": "aedff9dc-e4fa-473d-c3b2-67924262af3d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'temperature': 0.4, 'max_output_tokens': 512}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "### Step -15 #### Conversational Bot\n",
        "chat_history = []\n",
        "\n",
        "def chat_with_bot(user_input):\n",
        "    global chat_history\n",
        "\n",
        "    context = retrive_context(user_input)\n",
        "    prompt = f\"\"\"\n",
        "    You are a helpful conversational data analyst assistant. Please refer to the context below and answer the user's question.\n",
        "    Context:\n",
        "    {context}\n",
        "    User's Question:\n",
        "    {user_input}\n",
        "\n",
        "    Rules:\n",
        "    - Be Conversational\n",
        "    - Answer only using the context\n",
        "    - If you don't know the answer, say you don't have enough information\n",
        "    \"\"\"\n",
        "    response = gemini_model.generate_content(prompt)\n",
        "    answer = response.text\n",
        "    chat_history.append({\"user\": user_input, \"bot\": answer})\n",
        "    return answer"
      ],
      "metadata": {
        "id": "W-R5MapOLm_x"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#### Final Step ### Step -16\n",
        "print(\"Order Analytics Chat Bot Ready !!!\")\n",
        "print(\"Type 'exit' to stop\\n\")\n",
        "\n",
        "while True:\n",
        "  user_input = input(\"User: \")\n",
        "  if user_input.lower() in ['exit', 'quit', 'bye']:\n",
        "    print(\"Good Bye !!!\")\n",
        "    break\n",
        "  response = chat_with_bot(user_input)\n",
        "  print(f\"Bot: {response}\")\n",
        "  print(\"-\"*60)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 208
        },
        "id": "W_KJ62gRO2sE",
        "outputId": "9bac7fa6-560d-4ff6-ca9b-199df08cca7e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Order Analytics Chat Bot Ready !!!\n",
            "Type 'exit' to stop\n",
            "\n",
            "User: what is total sales\n",
            "Bot: Based on the information provided, the total sales amount is 2,528,828.\n",
            "------------------------------------------------------------\n",
            "User: which customer has the highest sales ?\n",
            "Bot: Based on the information provided, Jim Radford has the highest sales with a total of 1,669,808.\n",
            "------------------------------------------------------------\n",
            "User: quit\n",
            "Good Bye !!!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## PDF Extraction ----> Converted that to Dataframe ----> Converted to Text\n",
        "### --- Created the Embeddings ----> Stored to Vector DB ---- > Genererated Context ----- >\n",
        "#### --- Created the Model Configuration --- > Prompt Engineering ---- > Creating the Conversational Bot\n",
        "#### Project Solution Architecture\n",
        "\n",
        "#### What is API ?\n",
        "### What is Pre-trained models ? Examples ?\n",
        "### I have to built a gen AI application to support my sales team ? what kind of applications I can built to improve sales perf.\n",
        "### What is LLM Models ?\n",
        "### What is a prompt and why does prompt quality matter ?\n",
        "### What are tokens in language models ?\n",
        "### Tell me about different embeddings models ?\n",
        "### How do transformers work ?\n",
        "### What is RAG ? How do you use it in your current project ?\n",
        "### What are multi-modals ?\n",
        "### What are the challenges in GEN AI ?\n",
        "\n",
        "\n",
        "### Next Step - Agentic AI // Langchain (Framework) ### Copilot   ### Bedrock (AWS)\n",
        "### Create chatbots on any data using RAG and LLM ### # 3-4 indutry vertical put in resume\n",
        "###"
      ],
      "metadata": {
        "id": "gO1Je4SURY4b"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}