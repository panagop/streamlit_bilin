import numpy as np
import matplotlib.pyplot as plt
import IPython
import plotly
import json
import math


class Bilin:
    def __init__(self, xtarget=0.0, dropstrength=0.75, elastoplastic=False, allowa010=True):
        self.logtext = []
        self.xtarget = xtarget
        self.dropstrength = dropstrength
        self.elastoplastic = elastoplastic
        self.allowa010 = allowa010

        self.EPSILON = 0.0000001

    def load_space_delimited(self, fname, delimiter):
        np.loadtxt(fname)
        self.x_ini, self.y_ini = np.loadtxt(fname, delimiter=delimiter, usecols=(0, 1), unpack=True)

        self.logtext.append('initial X values')
        self.logtext.append(str(self.x_ini))
        self.logtext.append('')
        self.logtext.append('initial Y values')
        self.logtext.append(str(self.y_ini))

    @staticmethod
    def __curve_to_xcheck(x, y, xtarget):
        # Γραμμική παρεμβολή για να βρεθεί η τιμή του a στο d
        _ytarget = float(np.interp(xtarget, x, y))

        # Κρατώ μόνο τις τιμές μέχρι μία πρίν από το xtarget
        _x = x[(x < xtarget)]
        _y = y[0:len(_x)]

        # Προσθέτω τα x, y της παρεμβολής
        _x = np.append(_x, xtarget)
        _y = np.append(_y, _ytarget)

        return _x, _y

    @staticmethod
    def __get_area(x, y):
        return np.trapz(y, x, 0.01)

    def calc(self):
        # Αυτά μπορεί να μη χρειαστούν
        _rCount = len(self.y_ini)
        _maxY = max(self.y_ini)
        _minY = min(self.y_ini)
        _maxX = max(self.x_ini)
        _minX = min(self.x_ini)

        # Βρίσκω την αρχική μετατόπιση σε περίπτωση που δεν είναι 0
        x_ini0 = self.x_ini[0]

        self.logtext.append('')
        self.logtext.append(f'Αρχική μετατόπιση: {x_ini0}')

        # Κρατώ την καμπύλη μέχρι το xtarget, αν υπάρχει
        if self.xtarget > 0:
            x_xcheck, y_xcheck = self.__curve_to_xcheck(self.x_ini, self.y_ini, self.xtarget)
        else:
            x_xcheck = self.x_ini
            y_xcheck = self.y_ini

        # Αφαιρώ την αρχική μετατόπιση ώστε η καμπύλη να ξεκινά από το (0, 0)
        x_xcheck = x_xcheck - x_ini0

        self.logtext.append('')
        self.logtext.append('X values μέχρι το xtarget, αφαιρώντας (αν υπάρχει) την αρχική μετατόπιση')
        self.logtext.append(str(x_xcheck))
        self.logtext.append('')
        self.logtext.append('Y values μέχρι το xtarget')
        self.logtext.append(str(y_xcheck))

        # Βρίσκω τις δυσκαμψίες σε κάθε βήμα
        # Αλλάχω προσωρινά το x(0) για να μη διαιρεί με 0
        x_xcheck[0] = self.EPSILON
        k = np.divide(y_xcheck, x_xcheck)
        x_xcheck[0] = 0.0

        self.logtext.append('')
        self.logtext.append('Δυσκαμψίες (y(i)/x(i)')
        self.logtext.append(str(k))

        y02 = 0.2 * max(y_xcheck)
        x02 = float(np.interp(y02, y_xcheck, x_xcheck))

        k02 = y02 / x02

        self.logtext.append('')
        self.logtext.append('Έλεγχος στο 20% του ymax')
        self.logtext.append(f'x(02)={x02}, y(02)={y02}. Οπότε k(02)={k02}')

        # Βρίσκω το εμβαρό
        area = self.__get_area(x_xcheck, y_xcheck)
        self.logtext.append('')
        self.logtext.append(f'Εμβαδό καμπύλης: {area}')

        count = 0
        kel = k02
        # while np.abs((kel - k_06) / k_06) > 0.00000001:
        while True:
            count += 1  # This is the same as count = count + 1

            x_y, y_y, x_u, y_u, kinel, k_06 = self.iteration(x_xcheck, y_xcheck, kel, area)
            self.logtext.append('')
            self.logtext.append(f'iteration: {count}')
            self.logtext.append(f'x_y= {x_y}')
            self.logtext.append(f'y_y= {y_y}')
            self.logtext.append(f'x_u= {x_u}')
            self.logtext.append(f'y_u= {y_u}')
            self.logtext.append(f'kinel= {kinel}')
            self.logtext.append(f'kel= {kel}')
            self.logtext.append(f'k_06= {k_06}')

            if(np.abs((kel - k_06) / k_06) < 0.000001):
                break
            else:
                kel = k_06



            # return x_y, y_y, x_u, y_u, kinel, k_06

    def iteration(self, x, y, kel, area):
        rcount = len(x) - 1
        ymax = max(y)

        # ********** Αρχικός υπολογισμός *****************
        if y[rcount - 1] >= (2 * ymax + y[rcount]) / 3.0 and y[rcount] <= (2 * ymax + y[rcount]) / 3.0:
            y_u = y[rcount - 1]
        else:
            y_u = (2 * ymax + y[rcount]) / 3.0

        x_u = x[rcount]

        x_y = (2 * area - x[rcount] * y_u) / (kel * x[rcount] - y_u)
        y_y = kel * x_y

        kinel = (y_u - y_y) / (x_u - x_y)

        y_06 = 0.6 * y_y
        x_06 = float(np.interp(y_06, y, x))
        k_06 = y_06 / x_06

        return x_y, y_y, x_u, y_u, kinel, k_06

    def __str__(self):
        return ('\n').join((self.logtext))


bl = Bilin(xtarget = 0.00)

bl.load_space_delimited(r'D:\Programming\PyMyPackages\pystreng\tests\bilin\Example8', ' ')
# bl.set_options(0.06)
bl.calc()


print(bl)