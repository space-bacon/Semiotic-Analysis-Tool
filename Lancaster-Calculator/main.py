from lancaster_logic import compute_codex_psi

def main():
    chi = float(input("Enter χ (Cross-Event Index): "))
    C = float(input("Enter C (Consciousness Load): "))
    S = float(input("Enter S (Silence Potential): "))

    psi = compute_codex_psi(chi, C, S)
    print(f"\nCodex Reality Quotient Ψ = {psi}")

if __name__ == "__main__":
    main()
