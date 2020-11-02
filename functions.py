def strBool(s):
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    return bool(s)

if __name__ == "__main__":
    print(strBool("false"))