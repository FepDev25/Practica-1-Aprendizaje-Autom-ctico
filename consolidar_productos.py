#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para consolidar todos los productos de las diferentes categorías
en un archivo principal productos.txt

Autor: Generado automáticamente
Fecha: 15 de octubre de 2025
"""

import os
from pathlib import Path

def consolidar_productos():
    """
    Lee todos los archivos .txt de la carpeta 'productos/' y los consolida
    en el archivo principal 'productos.txt'
    """
    
    # Definir rutas
    carpeta_base = Path(__file__).parent
    carpeta_productos = carpeta_base / 'productos'
    archivo_salida = carpeta_base / 'productos.txt'
    
    # Verificar que existe la carpeta productos
    if not carpeta_productos.exists():
        print(f"❌ Error: No se encuentra la carpeta '{carpeta_productos}'")
        return
    
    # Obtener todos los archivos .txt de la carpeta productos
    archivos_categorias = sorted(carpeta_productos.glob('*.txt'))
    
    if not archivos_categorias:
        print(f"⚠️  Advertencia: No se encontraron archivos .txt en '{carpeta_productos}'")
        return
    
    print(f"📁 Encontrados {len(archivos_categorias)} archivos de categorías")
    print("="*70)
    
    # Abrir el archivo de salida en modo escritura
    with open(archivo_salida, 'w', encoding='utf-8') as archivo_principal:
        
        total_productos = 0
        
        # Procesar cada archivo de categoría
        for archivo_categoria in archivos_categorias:
            
            # Obtener nombre de la categoría (sin extensión)
            nombre_categoria = archivo_categoria.stem.replace('_', ' ').title()
            
            print(f"📋 Procesando: {nombre_categoria}")
            
            # Leer contenido del archivo
            try:
                with open(archivo_categoria, 'r', encoding='utf-8') as f:
                    lineas = f.readlines()
                
                # Filtrar líneas vacías y contar productos
                productos = [linea.strip() for linea in lineas if linea.strip()]
                num_productos = len(productos)
                total_productos += num_productos
                
                # Escribir productos directamente, sin numeración ni categoría
                for producto in productos:
                    archivo_principal.write(f"{producto}\n")
                
                print(f"   ✓ {num_productos} productos agregados")
                
            except Exception as e:
                print(f"   ❌ Error al procesar '{archivo_categoria.name}': {e}")
                continue
    
    print("\n" + "="*70)
    print("✅ CONSOLIDACIÓN COMPLETADA")
    print("="*70)
    print(f"📊 Total de categorías: {len(archivos_categorias)}")
    print(f"📦 Total de productos: {total_productos}")
    print(f"📄 Archivo generado: {archivo_salida}")
    print("="*70)


def listar_categorias():
    """
    Lista todas las categorías disponibles sin consolidar
    """
    carpeta_base = Path(__file__).parent
    carpeta_productos = carpeta_base / 'productos'
    
    archivos_categorias = sorted(carpeta_productos.glob('*.txt'))
    
    print("\n📂 CATEGORÍAS DISPONIBLES:")
    print("="*70)
    for i, archivo in enumerate(archivos_categorias, 1):
        nombre = archivo.stem.replace('_', ' ').title()
        print(f"{i:2d}. {nombre:<40} ({archivo.name})")
    print("="*70)
    print(f"Total: {len(archivos_categorias)} categorías\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONSOLIDADOR DE PRODUCTOS")
    print("="*70 + "\n")
    
    # Listar categorías disponibles
    listar_categorias()
    
    # Preguntar al usuario si desea continuar
    respuesta = input("¿Desea consolidar todos los productos en 'productos.txt'? (s/n): ")
    
    if respuesta.lower() in ['s', 'si', 'sí', 'yes', 'y']:
        print("\n🔄 Iniciando consolidación...\n")
        consolidar_productos()
    else:
        print("\n❌ Operación cancelada por el usuario.")
