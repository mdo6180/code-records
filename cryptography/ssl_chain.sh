#!/bin/bash
# adapted from https://kdecherf.com/blog/2015/04/10/show-the-certificate-chain-of-a-local-x509-file/

# to edit this file, do the following:
# $ sudo su
# $ vim ssl_chain.sh

# to run:
# $ sudo su
# $ ssl_chain.sh filename.crt

chain_pem="${1}"

if [[ ! -f "${chain_pem}" ]]; then
    echo "Usage: $0 BASE64_CERTIFICATE_CHAIN_FILE" >&2
    exit 1
fi

if ! openssl x509 -in "${chain_pem}" -noout 2>/dev/null ; then
    echo "${chain_pem} is not a certificate" >&2
    exit 1
fi

echo
awk -F'\n' '
        BEGIN {
            showcert = "openssl x509 -noout -subject -issuer -ext 'subjectKeyIdentifier,authorityKeyIdentifier'"
        }

        /-----BEGIN CERTIFICATE-----/ {
            printf "%2d: ", ind
        }

        {
            printf $0"\n" | showcert
        }

        /-----END CERTIFICATE-----/ {
            close(showcert)
            ind ++
	    printf "\n"
        }
    ' "${chain_pem}"

echo
openssl verify -untrusted "${chain_pem}" "${chain_pem}"
echo
