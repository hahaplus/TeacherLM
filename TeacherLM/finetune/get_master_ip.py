import sys


def get_master_ip(orgip):
    if orgip.strip() == "":
        print("localhost")
        return "localhost"
    flag = -1
    resip = ""
    if "[" not in orgip and "]" not in orgip:
        print("localhost")
    for i, ss in enumerate(orgip):
        if ss == "[":
            flag = i
            resip = orgip[:i]
        else:
            if flag != -1 and (ss == "," or ss == "-"):
                resip += orgip[flag + 1 : i]
                break
    print(resip)
    return resip


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1].strip() == "":
        print("localhost")
        exit(0)
    get_master_ip(sys.argv[1])
