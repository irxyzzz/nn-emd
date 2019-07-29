
from charm.toolbox.ecgroup import ECGroup
from charm.toolbox.eccurve import prime192v2
from charm.toolbox.integergroup import IntegerGroupQ, integer
from crypto.pke import PKEElGamal


def test_pke_elgamal85():
    # groupObj = ECGroup(prime192v2)
    # pke_elgamal = PKEElGamal(groupObj)
    # (public_key, secret_key) = pke_elgamal.keygen()
    # msg = b"hello world!12345678"
    # cipher_text = pke_elgamal.encrypt(public_key, msg)
    # decrypted_msg = pke_elgamal.decrypt(public_key, secret_key, cipher_text)
    # print(str(decrypted_msg))

    # assigned p and q here; otherwise, it will be generated and cost some time.
    p = integer(148829018183496626261556856344710600327516732500226144177322012998064772051982752493460332138204351040296264880017943408846937646702376203733370973197019636813306480144595809796154634625021213611577190781215296823124523899584781302512549499802030946698512327294159881907114777803654670044046376468983244647367)
    q = integer(74414509091748313130778428172355300163758366250113072088661006499032386025991376246730166069102175520148132440008971704423468823351188101866685486598509818406653240072297904898077317312510606805788595390607648411562261949792390651256274749901015473349256163647079940953557388901827335022023188234491622323683)
    groupObjInteger = IntegerGroupQ()
    pke_elgamal_int = PKEElGamal(groupObjInteger, p, q)
    (public_key, secret_key) = pke_elgamal_int.setup()
    msg = b"atestasdd hello world!222"
    cipher_text = pke_elgamal_int.encrypt(public_key, msg)
    decrypted_msg = pke_elgamal_int.decrypt(public_key, secret_key, cipher_text)
    print(str(decrypted_msg))

test_pke_elgamal85()