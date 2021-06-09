import os
import shutil
from docutils.core import publish_doctree

from scopesim.reports import rst_utils as ru
from scopesim.tests.mocks.py_objects import report_objects as ro
from scopesim import rc
rc.__config__["!SIM.reports.image_path"] = "./images_temp/"
rc.__config__["!SIM.reports.latex_path"] = "./latex_temp/"
rc.__config__["!SIM.reports.rst_path"] = "./rst_temp/"

CLEAN_UP = True
PLOTS = False

IMG_PATH = rc.__config__["!SIM.reports.image_path"]
LATEX_PATH = rc.__config__["!SIM.reports.latex_path"]
RST_PATH = rc.__config__["!SIM.reports.rst_path"]


def setup_module():
    for path in [IMG_PATH, LATEX_PATH, RST_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)
    rc.__config__["!SIM.reports.image_path"] = IMG_PATH
    rc.__config__["!SIM.reports.latex_path"] = LATEX_PATH
    rc.__config__["!SIM.reports.rst_path"] = RST_PATH


def teardown_module():
    for path in [IMG_PATH, LATEX_PATH, RST_PATH]:
        if CLEAN_UP:
            shutil.rmtree(path)


class TestWalk:
    def test_context_code_is_empty_with_reset_flag(self):
        doctree = publish_doctree(ro.reset_comment_snippet)
        code = ru.walk(doctree)
        assert code == ""


class TestPlotRstText:
    def test_image_file_exists_for_comment_node(self):
        assert os.path.exists(IMG_PATH)
        ru.plotify_rst_text(ro.comment_plot_snippet)
        assert os.path.exists(os.path.join(IMG_PATH, "my_fug.png"))
        assert os.path.exists(os.path.join(IMG_PATH, "my_fug.pdf"))

    def test_image_file_exists_for_literal_node(self):
        print(IMG_PATH)
        ru.plotify_rst_text(ro.literal_plot_snippet)
        assert os.path.exists(os.path.join(IMG_PATH, "my_fug3.svg"))
        assert os.path.exists(os.path.join(IMG_PATH, "my_fug3.png"))


class TestLatexifyRstText:
    def test_stuff(self):
        ru.latexify_rst_text(ro.big_rst_text)
        assert os.path.exists(os.path.join(LATEX_PATH,
                                           "This_parrot_goes_vrooom.tex"))


class TestRstifyRstText:
    def test_stuff(self):
        ru.rstify_rst_text(ro.big_rst_text)
        assert os.path.exists(os.path.join(RST_PATH,
                                           "This_parrot_goes_vrooom.rst"))


class TestPlotifyRstText:
    def test_stuff(self):
        print(IMG_PATH)
        ru.plotify_rst_text(ro.big_rst_text)
        fnames = ["my_fug_A.pdf", "my_fug_B.svg", "my_fug_c.png"]
        for fname in fnames:
            assert os.path.exists(os.path.join(IMG_PATH, fname))


class TestEffectReport:
    def test_all_parts_are_created_in_rc_folders(self):
        from scopesim.tests.mocks.py_objects import effects_objects as eo
        det_list = eo._detector_list()
        rst_text = det_list.report()
        ru.rstify_rst_text(rst_text, title_char="*", filename=det_list.meta["name"])
        ru.latexify_rst_text(rst_text, title_char="*", filename=det_list.meta["name"])
