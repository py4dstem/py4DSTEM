from ncem_4D_stem_quickview import Interactive4DSTEMDataViewer
import sys

if __name__ == '__main__':
    app = Interactive4DSTEMDataViewer(sys.argv)

    sys.exit(app.exec_())
