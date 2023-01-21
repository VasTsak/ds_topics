import logging

year = range(2022, 2023)
#month = range(1, 13)
month = range(1, 4)
day = range(1, 29)
hour = range(24)
minute = range(60)
second = range(0, 30, 5)

logging.basicConfig(filename='app.log', filemode='w')

def format_log(message, *args):
    date = ""
    for a in args:
        date += str(a) + ":"
    date += ":"
    message = date + message
    return message


def main():
        
    for y in year:
        for m in month:
            for d in day:
                for h in hour:
                    for m in minute:
                        for s in second:
                            message = 'This is a debug message about network'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.debug(message)
                            message = 'This is a debug message about configuration'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.debug(message)
                            message = 'This is an info message about network'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.info(message)
                            message = 'This is an info message about configuration'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.info(message)
                            message = 'This is a warning message about insufficient memory'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.warning(message)
                            message = 'This is a warning message about wrong model file'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.warning(message)
                            message = 'This is an error message, model file not found'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.error(message)
                            message = 'This is an error message, data file not found'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.error(message)
                            message = 'This is an critical message about model not found'
                            message = format_log(message, y, m, d, h, m, s)
                            logging.critical(message)


main()