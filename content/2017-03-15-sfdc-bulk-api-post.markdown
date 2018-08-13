Title: A simple Salesforce.com bulk API client for Python 3
Date: 2017-03-15
Tags: sfdc, api_client, bulk_api, python3
Slug: sfdc-bulk-api

I recently put together [my first Python package](https://github.com/donaldrauscher/sfdc-bulk)!  My company (AbleTo) recently migrated our CRM to Salesforce.com.  Though not the most elegant interface, many tools (e.g. Marketo) provide off-the-shelf integrations with Salesforce, which was very appealing to us.  To migrate all of our customer data (several million records), I used Salesforce.com's [bulk API](https://developer.salesforce.com/docs/atlas.en-us.api_asynch.meta/api_asynch/asynch_api_intro.htm).  The big difference between working with the REST API and working with the bulk API is that the bulk API is *asynchronous*.  You create a job, add one or more batches to that job, close the job, and wait for it to complete.  Once complete, you download the results (in pieces).  So a single 'operation' requires several HTTP calls to the bulk API and a rather annoying process of stitching together results.  This library provides a few methods to simplify this workflow.  Inputs and outputs are Pandas dataframes.  To avoid reinventing the wheel, I leveraged the [SalesforceLogin](https://github.com/simple-salesforce/simple-salesforce/blob/master/simple_salesforce/login.py) class in the simple_salesforce package for authentication.

Here are the key methods for performing bulk API queries:

+ **query**: creates a query job and submits a SOQL query as a batch to that job
+ **get_all_query_results**: downloads results from a query job into a Pandas dataframe

Here are the key methods for performging bulk API jobs:

+ **create_job**: creates a job (operations types include 'insert', 'upsert
, 'update', 'delete', and 'hardDelete')
+ **bulk_csv_operation**: breaks Pandas dataframe into chunks and adds each chunk as a batches to a job
+ **get_bulk_csv_operation_results**: downloads results from a bulk CSV job into a Pandas dataframe

Here is the an example of how to use the library:
``` python
from sfdc_bulk import SalesforceBulkAPI

# set up our SF object
bulk = SalesforceBulkAPI(**sfdc_credentials)

# pull down some records into a pandas df
query_job = bulk.query('SELECT Id, Company FROM Lead LIMIT 10000')
some_records = bulk.get_all_query_results(query_job)

# make an update
update = some_records[some_records.Company == "Blockbuster Video"]
update['DoNotCall'] = 1

# push to SFDC
update_job = bulk.create_update_job(object='Lead', contentType='CSV')
bulk.bulk_csv_operation(update_job, update)
update_results = bulk.get_bulk_csv_operation_results(update_job)
```

Package is available on [PyPI](https://pypi.python.org/pypi/sfdc-bulk), so it can be installed easily with pip.  Enjoy!
