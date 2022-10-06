import base64

c = ""
with open("test1.jpg", "rb") as image2string:
    converted_string = base64.b64encode(image2string.read())
# print(c)
with open("base64Bin.bin","wb") as string2bin:
    string2bin.write(c)
with open("encode.bin", "wb") as file:
    file.write(converted_string)
