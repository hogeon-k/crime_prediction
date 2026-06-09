"""
GUI 실행 진입점 (run_gui.py)

Crime Generator GUI를 실행하는 스크립트입니다.
src/gui 패키지의 CrimeGeneratorWindow를 호출합니다.
"""

from gui.crime_generator_window import CrimeGeneratorWindow

if __name__ == "__main__":
    CrimeGeneratorWindow().run()
