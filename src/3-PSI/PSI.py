import random
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import hashlib

domain = {"p": 9223372036854771239, "q": 4611686018427385619}

def hash_numeric(input_data):
    if isinstance(input_data, str):
        input_str = input_data
    elif isinstance(input_data, torch.Tensor):
        input_str = str(input_data.numpy().tolist())
    else:
        input_str = str(input_data)
  
    bytes = input_str.encode('utf-8')
    
    hash_obj = hashlib.sha1()
    hash_obj.update(bytes)
    
    hash_hex = hash_obj.hexdigest()
    hash_int = int(hash_hex,16)
        
    return hash_int

def _encrypt(array, e):
    result = np.array([pow(int(x),e,domain["p"]) for x in array],dtype = np.uint64)
    return result

def create_key():
    return random.randint(1, domain["q"]-1)




if __name__ == "__main__":
    # Los elementos de VS y VR deben ser únicos
    #VS = ["48XX2XX72J", "2", "3", "4", "5"]
    #VR = ["3", "4", "5", "6", "48XX2XX72J"]
    class DNIDataset(Dataset):
        def __init__(self, datos_dni, otros_datos, etiquetas):
            self.datos_dni = datos_dni
            self.otros_datos = otros_datos
            self.etiquetas = etiquetas
            
        def __len__(self):
            return len(self.datos_dni)
        
        def __getitem__(self, idx):
            dni = self.datos_dni[idx]
            otro_dato = self.otros_datos[idx]
            etiqueta = self.etiquetas[idx]
            return dni, otro_dato, etiqueta

    # Ejemplo de uso
    datos_dni = ["12345678A", "87654321B", "54321678C", "67891234D"]
    otros_datos = ["datos1", "datos2", "datos3", "datos4"]
    etiquetas = torch.tensor([0, 1, 2, 3])
    dni_dataset_1 = DNIDataset(datos_dni, otros_datos, etiquetas)

    datos_dni = ["67891234D", "22222222Y" ,"54321678C", "11111111X"]
    otros_datos = ["datos5", "datos6", "datos7", "datos8"]
    etiquetas = torch.tensor([3, 1, 5, 6])
    dni_dataset_2 = DNIDataset(datos_dni, otros_datos, etiquetas)

    VS = dni_dataset_1[:][0] # Obtengo lista DNI
    VR = dni_dataset_2[:][0]

   # Step 1: Hashing sets
    hash_size = hashlib.sha1().digest_size

    # Create an empty array to store the hashes
    XS = np.empty(len(dni_dataset_1), dtype=np.uint64)
    for i, sample in enumerate(dni_dataset_1):
        value = sample[0] # Cojo el DNI
        if isinstance(value, torch.Tensor):
            value = str(value.numpy())
        else:
            value = str(value)
        hash_value = int.from_bytes(hashlib.sha1(value.encode("utf-8")).digest(), byteorder="big")
        XS[i] = hash_value & 0xFFFFFFFFFFFFFFFF
    # XS.sort()
    
    XR = np.empty(len(dni_dataset_2), dtype=np.uint64)
    for i, sample in enumerate(dni_dataset_2):
        value = sample[0] # Cojo el DNI
        if isinstance(value, torch.Tensor):
            value = str(value.numpy())
        else:
            value = str(value)
        hash_value = int.from_bytes(hashlib.sha1(value.encode("utf-8")).digest(), byteorder="big")
        XR[i] = hash_value & 0xFFFFFFFFFFFFFFFF
    # XR.sort()

    print(f"XS = {XS}")
    print(f"XR = {XR}")
    
    # Step 2: Encrypting hashed sets
    eS = create_key()  # Random secret key for S
    eR = create_key()  # Random secret key for R
    # YS = encrypt_set(XS, eS)
    # YR = encrypt_set(XR, eR)
    YS = _encrypt(XS, eS)
    YR = _encrypt(XR, eR)

    print(f"eS = {eS}")
    print(f"eR = {eR}")
    
    print(f"YS = {YS}")
    print(f"YR = {YR}")
    
    # Step 3: Reordering lexicographically
    # YR_sorted = reorder_lexicographically(YR)
    # YS_sorted = reorder_lexicographically(YS)
    # Ahora, S tiene YR_sorted
    #        R tiene YS_sorted
    YR_sorted = np.sort(YR)
    YS_sorted = np.sort(YS)
    
    # Step 4b: Encrypting each element of YR with eS and sending back to R
    # encrypted_pairs_SR = encrypt_and_send_back(YR_sorted, eS) # (YR, Enc_s(YR))  esto se lo envia S a R 
    # encrypted_pairs_RS = encrypt_and_send_back(YS_sorted, eR) # (YS, Enc_r(YS)) esto se lo envia R a S
    Enc_s_YR = _encrypt(YR_sorted, eS)
    Enc_r_YS = _encrypt(YS_sorted, eR)
    
    encrypted_pairs_SR = np.column_stack((YR_sorted, Enc_s_YR))
    encrypted_pairs_RS = np.column_stack((YS_sorted, Enc_r_YS))
    print(f"Encrypted pairs que manda S a R = {encrypted_pairs_SR}")
    print(f"Encrypted pairs que manda R a S = {encrypted_pairs_RS}")   

    # Step 5: Decrypting and comparing sets
    f_eR_YS_sorted = _encrypt(YS_sorted, eR) # R encripta cada y in YS con eR, obteninendo ZS
    f_eS_YR_sorted = _encrypt(YR_sorted, eS)
    
    # Ahora R tiene ZS y encrypted_pairs
    # print(f"ZS = {f_eR_YS_sorted}")
    # Se coge solo los elementos de encrypted_pairs que estén en ZS:
    mask = np.isin(encrypted_pairs_SR[:,1], f_eR_YS_sorted)
    selected_pairs_R = encrypted_pairs_SR[mask]
    # selected_pairs_R = []
    # for pair in encrypted_pairs_SR:
    #     if pair[1] in f_eR_YS_sorted:
    #         selected_pairs_R.append(pair)
    mask = np.isin(encrypted_pairs_RS[:,1], f_eS_YR_sorted)
    selected_pairs_S = encrypted_pairs_RS[mask]
    # selected_pairs_S = []
    # for pair in encrypted_pairs_RS:
    #     if pair[1] in f_eS_YR_sorted:
    #         selected_pairs_S.append(pair)   
    

    # Obtener los primeros elementos de los pares en selected_pairs
    # selected_numbers_R = selected_pairs_R[:,0]
    # Obtener los índices de los elementos en YR que coinciden con los números seleccionados
    indices_R = np.where(np.isin(YR, selected_pairs_R[:,0]))[0]
    indices_S = np.where(np.isin(YS, selected_pairs_S[:,0]))[0]
   
    subset_R = Subset(dni_dataset_2, indices_R)
    subset_S = Subset(dni_dataset_1, indices_S)
    
    print(f"\nElementos en común según R:")
    for sample in subset_R:
        print(sample)
        
    print(f"\nElementos en común según S:")
    for sample in subset_S:
        print(sample)
    