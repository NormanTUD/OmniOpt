import socket
import os
from urllib.parse import urlparse
import mydebug
import re

'''
get_ip() returns the current local network ip if it can get it, if
not, it returns an empty string
'''
def get_ip():
    mydebug.debug('Trying to get IP')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    if ip:
        mydebug.debug('Obtained IP: ' + ip)
        return ip
    else:
        mydebug.debug('Could not get IP')
        return ''

def is_valid_ipv4_address(address):
    try:
        socket.inet_pton(socket.AF_INET, address)
    except AttributeError:  # no inet_pton here, sorry
        try:
            socket.inet_aton(address)
        except socket.error:
            return False
        return address.count('.') == 3
    except socket.error:  # not a valid address
        return False
    except Exception as e:
        return False

    return True

def is_valid_ipv6_address(address):
    try:
        socket.inet_pton(socket.AF_INET6, address)
    except socket.error:  # not a valid address
        return False
    except Exception as e:
        return False
    return True

def ping(server):
    if is_valid_ipv4_address(server) or is_valid_ipv6_address(server) or is_valid_url(server):
        response = os.system("ping -c 1 " + server + '> /dev/null')
        if response == 0:
            return True
        else:
            return False
    else:
        return False


def is_valid_url(server):
    if server == 'localhost':
        return True
    try:
        result = urlparse(server)
        regex = re.compile(
            r'(?:^(?:http|ftp)s?://)?' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if re.match(regex, server) is not None:
            return True
        #return True
    except ValueError:
        return False
    except:
        return False
    return False
