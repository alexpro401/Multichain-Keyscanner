import argparse
import ast
import asyncio
import binascii
# from lib.style import PrettyText
import datetime
import json
import os
# import sys
import time
from argparse import ArgumentParser
from pprint import pprint

# from moralis import evm_api
import aiofiles
import aiohttp
import dotenv
import web3
from aiohttp import ContentTypeError
from eth_utils import ValidationError
from tqdm.asyncio import tqdm
from web3.exceptions import ContractLogicError
from web3.middleware import geth_poa_middleware
from websockets.exceptions import ConnectionClosedError

import lib.abi_lib
from lib import style
from lib.ansi_colors import CBLUE2, CEND


try:
    from eth_utils.address import toChecksumAddress as to_checksum_address
except ImportError as err:
    from eth_utils.address import to_checksum_address
try:
    from eth_utils.conversions import toHex as to_hex
except ImportError:
    from eth_utils.conversions import to_hex
try:
    from eth_utils.curried import fromWei as from_wei
except ImportError:
    from eth_utils.curried import from_wei
try:
    from eth_utils.curried import toWei as to_wei
except ImportError:
    from eth_utils.curried import to_wei

_chain_map = {'ethereum': {'chain_id': 1, 'EvmChain': 'EvmChain.ETHEREUM'},
              'polygon': {'chain_id': 137, 'EvmChain': 'EvmChain.POLYGON'},
              'bnb': {'chain_id': 56, 'EvmChain': 'EvmChain.BSC'},
              'avalanche': {'chain_id': 43114, 'EvmChain': 'EvmChain.AVALANCHE'},  # C-Chain
              'fantom': {'chain_id': 250, 'EvmChain': 'EvmChain.FANTOM'},
              'cronos': {'chain_id': 25, 'EvmChain': 'EvmChain.CRONOS'},
              'palm': {'chain_id': 11297108109, 'EvmChain': 'EvmChain.PALM'},
              'arbitrum': {'chain_id': 42161, 'EvmChain': 'EvmChain.ARBITRUM'}}


class ScanSession:
    def __init__(self,
                 input_file: str = None,
                 output_file: str = None,
                 total_viable_wallets: int = 0,
                 total: float = 0,
                 report_dict: dict = {},
                 wallet_addresses: list = [],
                 token_map: list = [],
                 token_map_file: str = 'data/token.maps.json',
                 current_batch: int = 0,
                 total_gas: int = 0,
                 net: float = 0,
                 initialized=0,
                 event_loop=None,
                 chain='ethereum'):

        self._loop = event_loop
        if initialized == 0:
            self.initialized = time.time()
        else:
            self.initialized = initialized
        self.original_input_file = input_file
        self.network = chain
        self.input_file = f'{input_file}_{self.initialized}'
        self.output_file = output_file
        self.report_file = f'reports/{output_file}_{self.initialized}_{chain}'
        self.total_viable_wallets = total_viable_wallets
        self.total = total
        self.report_dict = report_dict
        self.wallet_addresses = wallet_addresses
        self.token_map = token_map
        self.token_map_file = token_map_file
        self.current_batch = current_batch
        self.total_gas = total_gas
        self.net = net
        # self.threshold= threshold
        self.token_map_ = {}
        self.session_file = None

    @property
    def __dict__(self):
        """
        Obtain the string representation of `Point`, so that just typing
        the instance name of an object of this type will call this method
        and obtain this string, just like `namedtuple` already does!
        """
        return {'input_file': self.input_file,
                'output_file': self.output_file, 'total_viable_wallets': self.total_viable_wallets,
                'total': self.total, 'report_dict': self.report_dict,
                'wallet_addresses': self.wallet_addresses, 'token_map': self.token_map,
                'token_map_file': self.token_map_file, 'current_batch': self.current_batch,
                'total_gas': self.total_gas, 'net': self.net, "initialized": self.initialized
                }

    def load_session(self, session_file):
        self.session_file = session_file
        with open(f'sessions/{session_file}', 'r') as f:
            _session = json.loads(json.dumps(f.read()))
            _session = ast.literal_eval(_session)
            for key, value in _session.items():
                setattr(self, key, value)

    async def dump_session(self):
        async with aiofiles.open(f'sessions/{self.session_file}', 'w') as f:
            await f.write(json.dumps(self.__dict__))


class ScanReport:
    """
    Initialize at scanner start, and updates as scanner goes.
    For each wallet initialize with address, key, ether value
    Then for each token in each wallet, update that entry
    Finally, calculate all of the totals at the end of the scan
    Last step = write the report to disc as json
    """

    def __init__(self, loop=None, network='ethereum', recalculate_zero_gas=False,
                 threshold=1.0):
        self._loop = loop
        self.network = network
        self.session = MultiChainScanner.session
        self.threshold = threshold
        self.recalculate_zero_gas = recalculate_zero_gas

    async def add_wallet(self, address: str, key: str = None, ether: float = 0.0, chain: str = None):
        """"
        new wallet for report
        """
        if not key:
            key = MultiChainScanner.load_keys(MultiChainScanner.session.original_input_file, find=address)
        eth_value = float(ether) * float(MultiChainScanner.eth_price)
        json_str = {'scanned_at': datetime.datetime.now().__str__(), 'private_key': key, 'address': address,
                    'eth': ether, 'tokens': [], 'total': eth_value, 'chain': chain}
        if ether > 0:
            pprint(json_str)
        self.session.report_dict[address] = json_str
        self.session.wallet_addresses.append(address)
        await self.async_dump_report()
        await self.session.dump_session()

    async def add_token(self, address, token_address, symbol, balance, usd_value, gas, price=0.0,
                        movable=False, swappable=False):
        """
        new token for existing wallet
        :param swappable: token can be transfered without revert
        :param movable: token is listed in a dex
        :param gas: the gas cost to transfer this token
        :param price: the price of this token
        :param address: wallet address
        :param token_address: contract address
        :param symbol: token symbol
        :param balance: token balance
        :param usd_value: us dollar value
        :return: bool
        """
        token_address = to_checksum_address(token_address)
        map_entry = {
            'contract': token_address,
            'symbol': symbol,
            'gas': gas}
        if self.session.token_map_.__contains__(token_address):
            pass
        else:
            await self.append_token_map(map_entry)

        entry = {'contract': token_address,
                 'balance': balance,
                 'usd_value': usd_value,
                 'price': price,
                 'symbol': symbol,
                 'gas': gas,
                 'movable': movable,
                 'swappable': swappable}
        try:
            self.session.report_dict[address]['tokens'].append(entry)
        except IndexError:
            MultiChainScanner.printer.error(f'ERROR: wallet: {address} not present!')
            return False
        else:
            if float(usd_value) >= self.threshold:
                self.session.net += usd_value
                self.session.total_gas = + int(gas)
                if movable and swappable:
                    self.session.total += usd_value
                    pprint(entry)
                self.session.total_viable_wallets += 1

            await self.async_dump_report()
            await self.session.dump_session()
        return True

    def load_token_map(self):
        self.session = MultiChainScanner.session
        MultiChainScanner.printer.normal(f'Loading token mappings from {self.session.token_map_file}... ')

        with open(MultiChainScanner.session.token_map_file, 'r') as fm:
            entries = [json.loads(json.dumps(line)) for line in fm.readlines()]
            self.session.token_map = sorted(set(entries))
        MultiChainScanner.printer.good(f'Loaded {len(entries)} tokens ... ')
        if len(entries):
            for tok in entries:
                # print(tok)
                tok = ast.literal_eval(tok)
                tok_addr = tok.get('contract')
                self.session.token_map_[tok_addr] = tok

    async def append_token_map(self, entry: dict):
        async with aiofiles.open(self.session.token_map_file, 'a') as mf:
            await mf.write(json.dumps(entry) + '\n')
        tok_addr = entry.get('contract')
        self.session.token_map_[tok_addr] = entry
        await asyncio.sleep(0)

    async def query_token_map(self, contract_address):
        if self.session.token_map_.__contains__(contract_address):
            return self.session.token_map_.get(contract_address)
        return False

    async def calculate_wallets(self, eth_price):
        """
        Tally up the results
        :param eth_price:
        :return:
        """
        # final_report = {}
        for address in self.session.wallet_addresses:
            total = 0
            tokens = self.session.report_dict.get(address).get('tokens')
            eth = float(self.session.report_dict.get(address).get('eth'))
            eth_usd = float(eth_price) * float(eth)
            for token in tokens:
                total += token.get('usd_value')
            self.session.report_dict.get(address)['total'] = total

            if float(total) > 0 or float(eth_usd) > 0:
                MultiChainScanner.printer.good(f'Address:{address}, Total: {total}, Eth: {eth} EthUsd: {eth_usd}')

            # self.session.total += total + eth_usd

        await asyncio.sleep(.25)
        rep_copy = self.session.report_dict.copy()
        for k, v in rep_copy.items():
            if v['total'] == 0:
                self.session.report_dict.pop(k)
        self.write()

    def write(self):
        """
        Dump the json report to disc
        :return:
        """
        with open(self.session.output_file, 'w') as f:
            json.dump(self.session.report_dict, f)

    async def async_load_report(self):
        async with aiofiles.open(self.session.output_file, 'r') as If:
            _report = json.loads(await If.read())

    async def async_dump_report(self):
        async with aiofiles.open(self.session.output_file, 'w') as of:
            await of.write(json.dumps(self.session.report_dict))

    def display(self):
        """
        pretty print the report
        :return:
        """
        pprint(self.session.report_dict)

    def total_all(self):
        """
        print total dollar value
        :return:
        """
        MultiChainScanner.printer.good(f'Total in all: {self.session.total}')


class MultiChainScanner:
    session = None
    eth_price = 0.0
    printer = style.PrettyText()

    def __init__(self, api_key=None,
                 delay=0.25,
                 log_file=None,
                 chain='ethereum',
                 batch=10,
                 batch_start_from=0,
                 verbosity=0,
                 event_loop=None,
                 w3_endpoint=None,
                 threshold=2):
        dotenv.load_dotenv()
        self.stats = {None}
        self.w3 = web3.Web3(web3.HTTPProvider(w3_endpoint))
        self.threshold = threshold
        self._loop = event_loop
        self.ws_connection = None
        self.api_key = api_key
        self.verbosity = verbosity
        self.delay = delay
        self.log_file = log_file
        self.batch = batch
        self.stats = {}
        self.blacklist = []
        self.api_key = os.environ.get('moralis_api_key')
        self.contract_map = {}
        self.batch_start_from = batch_start_from
        self._chains = []
        self.chain = chain
        self.scan_report = ScanReport(loop=self._loop, threshold=self.threshold)
        self.ws_connection = web3.Web3(web3.WebsocketProvider(os.environ.get(f'{self.chain}_ws_endpoint')))
        if chain == 'polygon':
            self.stable_contract = '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063'
            self.zrx_endpoint = 'https://polygon.api.0x.org/'
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        elif chain == 'arbitrum':
            self.stable_contract = '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1'
            self.zrx_endpoint = 'https://arbitrum.api.0x.org/'
        elif chain == 'optimism':
            self.stable_contract = '0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1'
            self.zrx_endpoint = 'https://optimism.api.0x.org'
        elif chain == 'bnb':
            self.stable_contract = '0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3'
            self.zrx_endpoint = 'https://bsc.api.0x.org/'
            #self.ws_connection = web3.Web3(web3.WebsocketProvider())
        else:
            # cs_wss = os.environ.get('chainstack_wss')
            # cs_user = os.environ.get('chainstack_user')
            # cs_pass = os.environ.get('chainstack_pass')
            # cs_ws_endpoint = f'wss://{cs_user}:{cs_pass}@{cs_wss}'

            self.stable_contract = '0x6B175474E89094C44Da98b954EedeAC495271d0F'
            self.zrx_endpoint = 'https://api.0x.org/'


        self._chains = [v.get('chain_id') for k, v in _chain_map.items()]
        self.headers = {"accept": "application/json", "X-API-Key": f"{self.api_key}"}
        self.endpoint = "https://deep-index.moralis.io/api/v2"
        self.result_dict = {}
        if not self.api_key:
            self.api_key = os.environ.get('moralis_api_key')

    async def __ainit__(self):
        self._session = aiohttp.ClientSession(headers=self.headers)
        self._session2 = aiohttp.ClientSession()
        self.scan_report.load_token_map()
        await self.get_native_price()
        await self.load_blacklist()

    async def __aclose__(self):
        await self._session.close()

    async def log_resp(self, text):
        async with aiofiles.open(self.log_file, 'a') as f:
            await f.write(str(text) + '\n')

    async def log_err(self, text):
        async with aiofiles.open('error_addresses', 'a') as f:
            await f.write(str(text) + '\n')

    async def get(self, url, *params):
        async with self._session.get(url=url, params=params) as response:
            status, resp = response.status, await response.json()
        return status, resp

    async def _get(self, *args):
        try:
            return await self.get(*args)
        except (asyncio.TimeoutError, ConnectionResetError) as err:
            if self.verbosity >= 1:
                self.printer.error('[!] Connection Error: %s' % err)
        except ContentTypeError as err:
            if self.verbosity >= 1:
                self.printer.error('[!] Invalid Response From API Error: %s' % err)
        return 400, {}

    async def post(self, url, data, *params):
        async with self._session.post(url=url, data=data, params=params) as response:
            status, resp = response.status, await response.json()
        return status, resp

    async def _get_wrapper(self, url, _name, _address):
        resp = await self._get(url)
        return {_name: (resp, _address)}

    async def get_scam_tokens(self):
        s, data = await self._get('https://raw.githubusercontent.com/dappradar/tokens-blacklist/main/all-tokens.json')
        if s == 200:
            with open('data/scam_tokens.json', 'w') as f:
                f.write(data)
        else:
            print(s, data)

    async def load_blacklist(self):
        self.printer.normal('Loading scam token list ... ')
        # await self.get_scam_tokens()
        with open('data/scam_tokens.json', 'r') as f:
            f = json.load(f)
        tokens = f['tokens']
        self.printer.normal(f'There are {len(tokens)} tokens in local blacklist ... ')
        for token in tokens:
            self.blacklist.append(token.get('address'))
        self.blacklist = f

    def query_blacklist(self, contract_address):
        if self.blacklist.__contains__(contract_address):
            return True
        return False

    @property
    def connection(self):
        if self.ws_connection:
            return self.ws_connection
        else:
            return self.w3

    async def get_native_price(self):
        self.printer.normal(f'Getting native {self.chain} price ... ')
        if self.chain == 'ethereum' or self.chain == 'arbitrum':
            s, r = await self.get(
                url=f'https://api.etherscan.io/api?module=stats&action=ethprice&apikey={os.environ.get("etherscan_api_key")}')
            __price = r.get('result').get('ethusd')
        elif self.chain == 'polygon':
            endpoint = 'https://api.polygonscan.com/'
            s, r = await self.get(url=endpoint + 'api' +
                                      f'?module=stats&action=maticprice&apikey={os.environ.get("polygonscan_api_key")}')
            # print(r)
            __price = r.get('result').get('maticusd')

        else:
            s, r = await self.get(
                f'https://api.bscscan.com/api?module=stats&action=bnbprice&apikey={os.environ.get("bscan_api_key")}')
            eth_price = r
            __price = eth_price.get('result').get('ethusd')
        self.eth_price = __price

    async def zrx_quote_all(self, contract, balance):
        if self.contract_map.get(contract):
            return self.contract_map.get(contract)
        else:
            self.contract_map[contract] = 0.0
            if contract == self.stable_contract:
                return 1.0
        params = {'sellToken': contract, 'buyToken': self.stable_contract, 'sellAmount': balance}
        resp = await self._session2.get(url=self.zrx_endpoint + 'swap/v1/quote', params=params)
        ret = await resp.json()
        if self.verbosity >= 3:
            self.printer.normal(f'ZRX: Status: {resp.status}')
        if resp.status == 200:
            p = float(ret.get('price'))
            self.contract_map[contract] = p
            return p
        else:
            return 0.0

    async def native_balance_query(self, addresses: str):
        tasks = []
        block = 0
        for address in addresses:
            for key, value in _chain_map.items():
                if self.chain == key:
                    cname = key
                    chain = value.get('chain_id')
                    url = f'{self.endpoint}/{address}/balance?chain={hex(chain)}&to_block={block}'
                    tasks.append(asyncio.ensure_future(self._get_wrapper(url, cname, address)))
        responses = await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)
        return responses

    async def token_balance_query(self, addresses: str):
        """
        Query the api and get the native asset balance

        :param addresses:
        :return:
        """
        tasks = []
        for address in addresses:
            for key, value in _chain_map.items():
                if self.chain == key:
                    cname = key
                    chain = value.get('chain_id')
                    url = f"{self.endpoint}/{address}/erc20?chain={hex(chain)}"
                    tasks.append(asyncio.ensure_future(self._get_wrapper(url, cname, address)))
        responses = await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)
        return responses

    def estimate_gas(self, contract_address, from_address, raw_balance):
        contract_address = to_checksum_address(contract_address)
        from_address = to_checksum_address(from_address)
        if self.chain == 'bnb':
            contract = self.connection.eth.contract(contract_address, abi=lib.abi_lib.BEP_ABI)
        else:
            contract = self.connection.eth.contract(contract_address, abi=lib.abi_lib.EIP20_ABI)
        try:
            est = contract.functions.transfer('0xb697d5e0324f3e7cd71a61526458dBfb8cF26035', raw_balance).estimate_gas(
                {'from': from_address})
        except (ContractLogicError, ValueError, ValidationError) as err:
            if self.verbosity >= 3:
                self.printer.warning(f'Gas estimation for contract: {contract_address} reverted with: {err}')
            return 0.0
        except (ConnectionClosedError, asyncio.exceptions.TimeoutError) as err:
            if self.verbosity >= 3:
                self.printer.error(f'Error with w3 connection: {err}')
            return 2
        else:
            return est

    def divide_chunks(self, l: list, n: int):
        """
        Split a list
        :param l: list
        :param n: batch size
        :return: generator
        """

        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    async def run(self, batch):
        """
        Process a batch of addresses.
        :param batch: list of addresses to scan
        :return:
        """
        native = await self.native_balance_query(batch)
        _native = []
        _token = []
        for res in native:
            if self.verbosity > 0:
                print(res)
            for chain, result in res.items():
                s = result[0][0]
                _data = result[1]

                if s == 200:
                    r = json.loads(json.dumps(result[0][1]))
                    r = int(r.get("balance"))

                    if r > 0:
                        actual_bal = from_wei(int(r), 'ether')
                        this = (chain, actual_bal)
                        _native.append(this)
                    else:
                        actual_bal = 0.0
                    await self.scan_report.add_wallet(address=_data, key=None, ether=int(actual_bal), chain=chain)

                else:
                    await self.log_err(_data)
        await asyncio.sleep(self.delay)
        tok = await self.token_balance_query(batch)
        for res in tok:
            if self.verbosity > 2:
                print(res)
            for chain, result in res.items():
                s = result[0][0]
                _data = result[0][1]

                if s == 200:
                    # print(result)
                    r = json.loads(json.dumps(result[0][1]))
                    # r = result[1]
                    if len(r) > 0:
                        this = (chain, result[1])
                        _address = result[1]
                        _token.append(this)
                        # _data = ast.literal_eval(_data)
                        for token in _data:
                            token = json.loads(json.dumps(token))
                            if token.get('decimals'):
                                decimals = int(token.get('decimals'))
                            else:
                                decimals = 0
                            raw_balance = int(token.get('balance'))
                            token_address = token.get('address')
                            is_scam = self.query_blacklist(token_address)
                            if not decimals or is_scam:
                                actual_token_balance = 0
                                price = 0
                                usd_value = 0
                                swappable = False
                                movable = False
                                gas_set = 0.0
                            else:
                                actual_token_balance = raw_balance / (10 ** int(decimals))
                                cached_token = await self.scan_report.query_token_map(token_address)
                                if cached_token:
                                    if self.session.token_map_.get('gas') == 0:
                                        price = 0.0
                                        pass
                                    else:
                                        price = await self.zrx_quote_all(token.get('token_address'),
                                                                         token.get('balance'))
                                else:
                                    price = await self.zrx_quote_all(token.get('token_address'), token.get('balance'))
                                usd_value = actual_token_balance * float(price)
                                if price:
                                    swappable = True
                                else:
                                    swappable = False

                                gas_set = self.estimate_gas(token.get('token_address'), _address, int(raw_balance))
                                if gas_set:
                                    movable = True
                                else:
                                    movable = False
                            # print(actual_token_balance, price, usd_value)
                            if movable and swappable:  # >= float(self.threshold)
                                await self.scan_report.add_token(_address, token.get('token_address'),
                                                                 token.get('symbol'),
                                                                 actual_token_balance,
                                                                 usd_value, gas_set, float(price), swappable=swappable,
                                                                 movable=movable)
                        await self.log_resp(text=(_data, this))
                else:
                    await self.log_err(_data)

    async def _process_batch(self, batch: list, progress: tqdm, start_time, skip=False):
        if skip:
            # progress.set_postfix_str("%s" % (CBLUE2 + str(self.stats) + CEND))
            return
        elapsed = time.time() - start_time
        self.stats = {'statistics': {'PNL': f'${self.session.total}',
                                     'Net': f'${self.session.total}',
                                     'Gw': self.session.total_gas}}
        await self.run(batch)
        progress.set_postfix_str("%s" % (CBLUE2 + str(self.stats) + CEND))
        self.session.current_batch += 1

    async def main(self, addresses: list):

        """
        Begin main logic. Split inputs into batches and start scanning each batch.
        If this is a restored session, seek ahead to last processed batch and resume.
        """
        self.printer.normal('Starting MultiCScan')
        batches = self.divide_chunks(addresses, self.batch)
        progress = tqdm(batches)
        total = len(addresses) / self.batch
        progress.total = total
        progress.colour = 'red'
        progress.set_description('Scan Progress')
        # progress.bar_format = '#'
        start = time.time()
        for x, batch in enumerate(progress):
            if self.batch_start_from != 0:
                if x < self.batch_start_from:
                    await self._process_batch(batch, progress, start, True)
                else:
                    await self._process_batch(batch, progress, start)
            else:
                await self._process_batch(batch, progress, start)
        await asyncio.sleep(0.1)
        end = time.time()
        self.printer.normal(f'Scan completed in {start - end} seconds!')

    @staticmethod
    def load_keys(file=None, find=None):
        """
        Load private keys into memory,
        or find a specific key corresponding
        to a particular address.
        :param file: list of private keys
        :param find: find the key that corresponds to this address
        :return:
        """
        if not file:
            return 'xxxxxxx'
        addresses = []
        with open(file, 'r') as f:

            f = f.readlines()
            for line in f:
                line = line.strip('\r\n')
                try:
                    acct = web3.eth.Account.from_key(line)
                    #acct = web3.eth.Eth.account.from_key(line)
                except (binascii.Error, ValueError):
                    try:
                        addr = to_checksum_address(line)
                    except ValueError:
                        pass
                    else:
                        addresses.append(addr)
                        if find:
                            if to_checksum_address(addr) == to_checksum_address(find.strip("\r\n")):
                                return 'xxxxxx'

                else:
                    addr = acct.address
                    # print(addr)
                    if find:
                        if to_checksum_address(addr) == to_checksum_address(find.strip("\r\n")):
                            return line

                    addresses.append(addr)
        return addresses

    async def finish(self):
        self.printer.normal('Generating final report ... ')
        await self.scan_report.calculate_wallets(MultiChainScanner.eth_price)
        self.scan_report.display()
        self.scan_report.write()

        self.printer.normal(f'Total viable wallets: {self.scan_report.session.total_viable_wallets}')
        self.printer.normal(f'Total gas needed to empty: {self.session.total_gas}')
        self.printer.normal(f'Total Net USD Value: ${self.session.total}')
        self.printer.normal(f'Total Profits: ${self.session.net}')

    async def start_loop(self, addresses):
        await self.__ainit__()
        await self.main(addresses)
        await self.finish()


def check_addr(address):
    if to_checksum_address(address):
        return address
    return False


def main():
    loop = asyncio.new_event_loop()
    args = argparse.ArgumentParser()
    args.add_argument('-f', '--file', dest='input_file', type=str, required=False,
                      help='List of private keys.')

    args.add_argument('-a', '--address', type=str)
    args.add_argument('-s', '--session', dest='session_file', type=str, default=None, help='Load a previous scan state')
    args.add_argument('-c', '--chain', type=str,
                      choices=['ethereum', 'polygon', 'bnb', 'avalanche', 'fantom', 'cronos', 'palm',
                               'arbitrum', 'optimism'], default='ethereum')
    args.add_argument('-b', '--batch_size', type=int, default=3)
    args.add_argument('-o', '--output', dest='output_file', type=str, default='multichain.log',
                      help='The log file of course.')
    args.add_argument('-D', '--delay', type=float, default=0.25,
                      help='Async sleep value.')
    args.add_argument('-t', '--threshold', type=float, default=2.00)
    args.add_argument('-fa', '--find_address', type=str, default=None, help='Given a list of keys, '
                                                                            'find the key corresponding'
                                                                            'to this address.')
    args.add_argument('-v', '--verbosity', dest='verbosity', default=0, action='count')
    args.add_argument('-d', '--debug', action='store_true', help='Enable various debugging features.')

    args = args.parse_args()
    dotenv.load_dotenv()
    w3_endpoint = os.environ.get('%s_http_endpoint' % args.chain)
    api = MultiChainScanner(None, args.delay, args.output_file,
                            chain=args.chain, batch=args.batch_size,
                            batch_start_from=0,
                            verbosity=args.verbosity, event_loop=loop,
                            w3_endpoint=w3_endpoint, threshold=args.threshold)

    if args.session_file:
        api.printer.good(f'Loading session file: {args.session_file}')

        scan_session = ScanSession(event_loop=loop, chain=args.chain, )
        scan_session.load_session(args.session_file)
        api.batch_start_from = scan_session.current_batch

    else:
        session_file = f'session_{time.time()}'
        api.printer.normal(f'Generated session file: {session_file}')
        scan_session = ScanSession(args.input_file, args.output_file, chain=args.chain, event_loop=loop, )
        scan_session.session_file = session_file
        api.batch_start_from = 0

    MultiChainScanner.session = scan_session

    chain = api.chain
    api.printer.good(f'Scanning network: {chain}')
    if args.input_file:
        addresses = api.load_keys(args.input_file)
    else:
        if args.address:
            addresses = [check_addr(args.address)]
            if addresses:
                print('[+] Valid ... ')
        else:
            addresses = []
    api.printer.good(f'Loaded {len(addresses)} addresses ... ')
    if args.find_address:
        api.printer.good(f'Found key: {api.load_keys(args.input_file, args.find_address)}')

    if args.debug:
        loop.run_until_complete(api.start_loop(addresses))
    else:
        try:
            loop.run_until_complete(api.start_loop(addresses))
        except OSError:
            pass
        except KeyboardInterrupt:
            api.printer.warning('[-] Caught Signal. Exit with grace.\n')
        finally:
            loop.run_until_complete(api.__aclose__())
            loop.close()
            exit(0)


if __name__ == '__main__':
    main()
