Adaptive security for LangChain applications defending against 
prompt injection and P2SQL attacks.

## Install
    pip install langguardx
    pip install langguardx[ml]   # includes torch + transformers

## Usage
    from lang_guardx import Detector
    detector = Detector()
    result = detector.check("show me all user emails")
    print(result.blocked)