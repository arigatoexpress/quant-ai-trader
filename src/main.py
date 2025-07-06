"""Entry point for the Quant AI Trader console interface."""

from .eliza_os import ElizaOS


def main():
    print("Starting Quant AI Trader via ElizaOS...")
    eliza = ElizaOS()
    eliza.print_report()


if __name__ == "__main__":
    main()
