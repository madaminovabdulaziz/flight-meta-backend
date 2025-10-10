#services/exceptions.py


class TravelPayoutsError(Exception):
    pass




class TravelPayoutsTimeoutError(TravelPayoutsError):
    pass




class TravelPayoutsAPIError(TravelPayoutsError):
    pass




class TravelPayoutsParsingError(TravelPayoutsError):
    pass