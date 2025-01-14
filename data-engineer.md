---
layout: default
title: Data Engineer
description: Data Engineer Projects
---

# Data Engineer Projects


## Title: Government Data Processing and Integration for India Data Portal

### Technology Stack:
- Web Scraping: BeautifulSoup (bs4), Selenium
- Data Processing: PySpark, Pandas
- Data Storage: Parquet file format
- Cloud Storage: Wasabi

### Problem Statement:
The project aimed to process and clean over 500GB of data sourced from various government websites, specifically focusing on datasets such as **MNREGA (Mahatma Gandhi National Rural Employment Guarantee Act) Physical, Financial, Mandays**, and others. The primary challenge was to efficiently gather, clean, and transform the data into a suitable format for integration with the India Data Portal.

### Steps Followed:
1. **Data Collection:**
   - Utilized web scraping techniques with BeautifulSoup (bs4) and Selenium to extract data from government websites.
  
2. **Data Processing:**
   - Leveraged PySpark for distributed data processing to handle the large dataset efficiently.
   - Used Pandas for certain data manipulation tasks, ensuring accuracy and ease of handling.

3. **Quality Check:**
   - Conducted a rigorous quality check by comparing processed data with the source to ensure accuracy and consistency.

4. **Data Transformation:**
   - Applied the necessary transformations, including converting data from long to wide format, aligning with the requirements of the India Data Portal.

5. **Data Storage:**
   - Converted the final processed data into the Parquet file format, optimizing storage and query performance.

6. **Cloud Storage and Accessibility:**
   - Uploaded the processed data to Wasabi, making it readily available and accessible for integration with the India Data Portal.

7. **Project Naming:**
   - The processed datasets were categorized into distinct project names, such as MNREGA Physical, MNREGA Financial, MNREGA Mandays, and others, facilitating easy identification and organization.

### Conclusion:
This project successfully addressed the challenges of handling massive government datasets, ensuring data accuracy, and transforming it into a format compatible with the India Data Portal. The use of a robust technology stack, including PySpark and Pandas, along with cloud storage on Wasabi, enabled efficient processing and accessibility of the valuable government data.


## Title:Flask API for LGD Data Processing

 **Technology Stack**
- Python (Flask, Pandas)

**Description**
Developed a Flask API for processing LGD-related JSON data in India. The API facilitates tasks such as processing LGD data, retrieving state mappings, and creating a mapped dataset based on user-defined inputs.

**Key Features**
- Efficient processing of LGD-related JSON data
- Retrieval of state mappings from an SQLite database
- Creation of mapped datasets based on provided inputs

**Endpoints**

**POST /process_json:**
- Process LGD-related JSON data and return a subset.

**GET /state_mappings:**
- Retrieve state mappings from the SQLite database.

**POST /create_mapped_dataset:**
- Create a mapped dataset based on user-defined inputs.

**Examples**
**Processing LGD Data**
- Send a POST request to http://localhost:5000/process_json with a JSON payload containing LGD-related data.

**Retrieving State Mappings**
- Send a GET request to http://localhost:5000/state_mappings.

**Creating Mapped Dataset**
- Send a POST request to http://localhost:5000/create_mapped_dataset with a JSON payload containing a 'dataset' and 'mapping'.


## Title: Import Export Data
- **Technology Stack:** Python, pandas, wasabi, ThreadPoolExecutor, boto3
- **Problem Statement:** Develop an ETL (Extract, Transform, Load) pipeline to extract import-export data from the Trade Statistics portal of the Ministry of Commerce and Industry, Government of India. Create a large dataset spanning 16 years with monthly trade data at the country level, totaling over 100 GB. Utilize Python programming language and pandas library for data manipulation and preparation during the ETL process.
- **Steps Followed:**
  - Applied ETL techniques to extract data from the Trade Statistics portal.
  - Created a large import-export dataset from scratch, spanning 16 years and consisting of over 100 GB of monthly trade data at the country level.
  - Used Python programming language and pandas library for data manipulation, cleaning, and preparation.
  - Executed several steps during the ETL process, including data extraction, cleaning, transformation, and loading.
  - Implemented the ThreadPoolExecutor and multiprocessing library to parallelize data processing tasks, significantly reducing execution time.
  - The resulting dataset is valuable for analyzing trade patterns, identifying market opportunities, and monitoring trade policy changes.
  - Successful completion of the project demonstrates skills in data manipulation, ETL processes, and handling large volumes of data.

## Title: Real-Time Stock Market Data with Kafka
- **Technology Stack:** Python, AWS, Apache Kafka, Glue, Athena, SQL
- **Problem Statement:** As a Data Engineer, develop a real-time stock market data system using Python and Apache Kafka. Implement data acquisition, processing, and analysis by leveraging AWS services like Glue and Athena. Enable real-time data computations and visualization using SQL queries and streaming frameworks. Ensure efficient and reliable data processing through performance optimization and error handling.
- **Project Overview:**
  - Developed a real-time stock market data system using Python and Apache Kafka.
  - Utilized AWS services like Glue and Athena for data acquisition, processing, and analysis.
  - Enabled real-time data computations and visualization through SQL queries and streaming frameworks.
  - Ensured efficient and reliable data processing by implementing performance optimization techniques and robust error handling.

## Title: MinDepExpScrapper
- **Technology Stack:** Python, pandas, Camelot
- **Problem Statement:** Create a data extraction and preprocessing pipeline using Python and Camelot to extract expenditure data from Ministries and Departments in PDF format. Preprocess the data using pandas and store it in CSV format. Perform data transformation by reshaping and aggregating the data to create a comprehensive and structured database.
- **Steps Followed:**
  - Used Camelot and pandas to extract expenditure data from Ministries and Departments in PDF format.
  - Preprocessed the extracted data using pandas and stored it in CSV format.
  - Performed data transformation by using pandas' melt and pivot functions to reshape the data into long and wide formats.
  - Aggregated the data to create a comprehensive and structured database.


## Title: PMAY Data Scrapper
- **Technology Stack:** Python, pandas, wasabi, ThreadPoolExecutor, selenium, boto3
- **Problem Statement:** As a Data Engineer for the PMAY Data Scrapper project, the goal was to automate the data acquisition and processing tasks for the Pradhan Mantri Awas Yojana (PMAY) using various Python libraries such as pandas, selenium, wasabi, ThreadPoolExecutor, and boto3. The responsibilities included automating the login process, extracting tokens, designing a logging mechanism, fetching data for specific cities, creating data frames, parallelizing data acquisition, converting data to CSV format, uploading files to S3, and following best practices for software development.
- **Steps Followed:**
  - Automated the login process and token extraction for accessing the PMAY data.
  - Designed and implemented a logging mechanism to track the data scraping process.
  - Developed data acquisition methods to fetch data for specific cities from the PMAY platform.
  - Utilized pandas library to create data frames for organizing and manipulating the acquired data.
  - Implemented ThreadPoolExecutor to parallelize data acquisition tasks, optimizing the data scraping process.
  - Converted the acquired data to CSV format for further analysis and storage.
  - Utilized boto3 library to upload the processed data files to the Wasabi storage platform.
  - Followed best practices for software development to ensure code quality and maintainability throughout the project.

[back](./)
