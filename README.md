# Multichain EVM Key Scanner

### About

<p>
This is a tool for scanning a large number of EVM accounts for balances. The tool takes a list of either private keys 
or addresses as input and creates a JSON report with all native and token balances. It also calculates how much gas 
is required in order to move those assets and calculates whether or not the value of the token is more than the required 
cost to move it. I wrote this because I needed to consolidate a bunch of change amounts of various tokens from some old 
wallets. This is one of a few versions that I wrote. It uses the Moralis API for token scanning and web3 for native 
balances. The following chains are supported:
</p>

<b> Supported Networks </b>

- Ethereum 
- Polygon
- Arbitrum
- BNB
- Optimism 
- Avalanche
- Cronos
- Palm
- Fantom
<p>

</p>

### Setup & Install

<p>
To install the tool, you simply need to create a new virtual enviroment (not strictly speaking necessary, but recommended) 
and install the requirements. You should be able to do that by running:
</p>
<pre>
$ virtualenv env
$ souce env/bin/activate
$ pip3 install -r requirements.txt
</pre>

<p>
Be aware that in some cases web3.py has issues with async support. If the program crashes seemingly inexplicably, 
run with --debug. If you get an error about the loop argument no longer being necessary with websockets, try reinstalling 
that lib with `pip3 install websockets --upgrade`. If that does not work, there is a version of web3 that will work, but 
I can't find the link at the moment. It's called web3 async filters.
</p>
<b>
Api Keys
</b>
<p>
In order to run this, you need the following API keys:
</p>

- [Moralis](https://moralis.io) -- required for getting token balances because web3 has no way to enumerate all possible tokens in a given wallet
- [Etherscan](https://etherscan.io) -- required to get the current ethereum price so we can calculate value and gas costs
- web3 , I recommend [Quicknode](https://quicknode.com)
  - HTTP endpoint
  - Websocket endpoint

<p>
Copy env.example to .env and add your api keys.
</p>


### Usage
<pre>
usage: main.py [-h] [-f INPUT_FILE] [-a ADDRESS] [-s SESSION_FILE] [-c {ethereum,polygon,bnb,avalanche,fantom,cronos,palm,arbitrum,optimism}] [-b BATCH_SIZE] [-o OUTPUT_FILE] [-D DELAY] [-t THRESHOLD] [-fa FIND_ADDRESS] [-v] [-d]

options:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --file INPUT_FILE
                        List of private keys.
  -a ADDRESS, --address ADDRESS
  -s SESSION_FILE, --session SESSION_FILE
                        Load a previous scan state
  -c {ethereum,polygon,bnb,avalanche,fantom,cronos,palm,arbitrum,optimism}, --chain {ethereum,polygon,bnb,avalanche,fantom,cronos,palm,arbitrum,optimism}
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -o OUTPUT_FILE, --output OUTPUT_FILE
                        The log file of course.
  -D DELAY, --delay DELAY
                        Async sleep value.
  -t THRESHOLD, --threshold THRESHOLD
  -fa FIND_ADDRESS, --find_address FIND_ADDRESS
                        Given a list of keys, find the key correspondingto this address.
  -v, --verbosity
  -d, --debug           Enable various debugging features.
                                                             
</pre>

<b>
Scam Token Detection Logic
</b>

<p>
Included in the `keys` directory is a list of example keys for demo purposes. These are keys that I took from the first 
couple of pages of the site https://keys.lol. As you can imagine, those keys are absolutely drowning in scam tokens. 
The program has logic to discern between scam tokens and tokens with actual value. As you can see in the asciinema below, 
the tool has managed to actually find some tokens of value in those publicly posted keys! It contains a blacklist of 
known scam tokens which it updates at launch. Additionally, the tool will simulate a transfer by calling `web3.eth.estimate_gas`,  
and if transfer fails, then the token is assumed to be a fake scam token and is not included in the report by default. 
Likewise, tokens that do not have any value are not included by default in the final report. However, 
there is an option to disable this if desired. 
</p>

#### Asciinema Demo

[![asciicast](https://asciinema.org/a/BYipIlrq7ictgLL9D90DvXcxt.svg)](https://asciinema.org/a/BYipIlrq7ictgLL9D90DvXcxt)


### Contact Me

<p>
If you find this program useful and would like to contact me, feel free to hit me up on telegram @chev603. If you are a 
business that has wallets with change that you would like to consolidate, contact me. I have a much more advanced, faster version 
of this code that also has logic to dump assets and can help you with that.
</p>