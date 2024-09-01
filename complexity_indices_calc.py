import numpy as np
import pandas as pd
from scipy import linalg


def input_country_products():
    country_products = {}
    all_products = set()

    while True:
        country = input("\nEnter a country name (or press Enter to finish): ").strip().lower()

        if not country:
            confirm = input("Are you sure you wish to finish adding countries (y/n)? ").strip().lower()
            if confirm == "y":
                break
            else:
                continue

        products = input(f"Enter products for {country} (comma-separated): ").strip().lower()
        product_list = [p.strip() for p in products.split(',')]
        
        country_products[country] = product_list
        all_products.update(product_list)
    
    return country_products, list(all_products)


def create_matrix(country_products, all_products):
    countries = list(country_products.keys())
    country_prod_mat = np.zeros((len(countries), len(all_products)))
    
    for i, country in enumerate(countries):
        for product in country_products[country]:
            j = all_products.index(product)
            country_prod_mat[i, j] = 1
    
    return country_prod_mat, countries, all_products


def calculate_complexity_metrics(country_prod_mat, countries, products):
    # Calculate diversity and ubiquity
    diversity = np.sum(country_prod_mat, axis=1)
    ubiquity = np.sum(country_prod_mat, axis=0)
    
    # Calculate matrices A_cp and B_cp
    country_diversity_mat = np.diag(1 / diversity)
    prod_ubiquity_mat = np.diag(1 / ubiquity)
    
    A_cp = country_diversity_mat @ country_prod_mat
    B_cp = country_prod_mat @ prod_ubiquity_mat
    
    # Calculate product space and country space
    prod_space_mat = A_cp.T @ B_cp
    country_space_mat = A_cp @ B_cp.T
    
    # Calculate ECI and PCI
    country_space_eigvals, country_space_eigvecs = linalg.eig(country_space_mat)
    
    # Sort eigenvalues and eigenvectors
    eig_idx = country_space_eigvals.argsort()[::-1]
    country_space_eigvals = country_space_eigvals[eig_idx]
    country_space_eigvecs = country_space_eigvecs[:, eig_idx]
    
    # ECI is the eigenvector corresponding to the second largest eigenvalue
    econ_complexity_index = pd.Series(country_space_eigvecs[:, 1].real, index=countries, name="ECI")
    
    # Calculate PCI
    prod_space_eigvals, prod_space_eigvecs = linalg.eig(prod_space_mat)
    
    # Sort eigenvalues and eigenvectors for products
    idx_p = prod_space_eigvals.argsort()[::-1]
    prod_space_eigvals = prod_space_eigvals[idx_p]
    prod_space_eigvecs = prod_space_eigvecs[:, idx_p]
    
    # PCI is the eigenvector corresponding to the second largest eigenvalue
    prod_complexity_index = pd.Series(prod_space_eigvecs[:, 1].real, index=products, name="PCI")
    
    return {
        "prod_space_mat": pd.DataFrame(prod_space_mat, index=products, columns=products),
        "country_space_mat": pd.DataFrame(country_space_mat, index=countries, columns=countries),
        "econ_complexity_index": econ_complexity_index,
        "prod_complexity_index": prod_complexity_index
    }


def main():
    print("Economic Complexity Calculator")
    print("==============================")
    
    # Input the country-product dictionary
    country_products, all_products = input_country_products()
    
    # Create the country-product matrix
    country_prod_mat, countries, products = create_matrix(country_products, all_products)
    
    print("\nCountry-Product Matrix:")
    print(pd.DataFrame(country_prod_mat, index=countries, columns=products))
    
    # Calculate complexity metrics
    results = calculate_complexity_metrics(country_prod_mat, countries, products)
    
    # Display results
    print("\nResults:")
    print("--------")
    
    print("\nProduct Space matrix:")
    print(results["prod_space_mat"])
    
    print("\nCountry Space matrix:")
    print(results["country_space_mat"])
    
    print("\nEconomic Complexity Index (ECI):")
    for country, eci in results["econ_complexity_index"].items():
        print(f"{country}: {eci}")
    
    print("\nProduct Complexity Index (PCI):")
    for prod, pci in results["prod_complexity_index"].items():
        print(f"{prod}: {pci}")

if __name__ == "__main__":
    main()