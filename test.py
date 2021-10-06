from datetime import datetime, timedelta
from json import detect_encoding
from Crypto.Cipher import AES
from Crypto.Util import Padding
import base64

end = (datetime.now() + timedelta(minutes=30)).timestamp()
key = b'7e6c8dcc8e1da2bd06946ec688de9553'
text = 'username@' + str(end)
print(text)

text = Padding.pad(text.encode(), 16)
print(text)

cipher = AES.new(key, AES.MODE_ECB)
en_text = cipher.encrypt(text)
print(en_text)

en_text = base64.encodebytes(en_text)
print(en_text)

en_text = en_text.decode()
print(en_text)

de_text = base64.decodebytes(en_text.encode())
print(de_text)
de_text = cipher.decrypt(de_text)
print(de_text)
de_text = Padding.unpad(de_text, 16)
print(de_text)
de_text = de_text.decode()
print(de_text)
