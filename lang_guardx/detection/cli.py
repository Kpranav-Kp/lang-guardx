def verify():
    try:
        import lang_guardx
        print(f"LangGuardX {lang_guardx.__version__} installed successfully.")
    except Exception as e:
        print(f"Installation failed: {e}")