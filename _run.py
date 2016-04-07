from ncem_4D_stem_quickview import NCEM4DSTEMQuickViewApp
import sys

if __name__ == '__main__':
    app = NCEM4DSTEMQuickViewApp(sys.argv)
    
    sys.exit(app.exec_())