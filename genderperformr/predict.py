from .genderperformr import GenderPerformr

gp = None

def predict(username):
    global gp
    if gp is None:
        try:
            gp = GenderPerformr()
        except Exception as e:
            print("Failed to initialize GenderPerformr")
            print(e)

    return gp.predict(username)


predict.__doc__ = "GenderPerformr Prediction Wrapper Function\n" + GenderPerformr.predict.__doc__
