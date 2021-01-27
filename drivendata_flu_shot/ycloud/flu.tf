provider "yandex" {
  service_account_key_file  = "/home/sergey/mnt/st1500/Usr/Sergey/TheJob/YandexCloud/Webvork/webvork-terraform-key.json"
  cloud_id  = "b1gbd722dh1ji2mfij3a"
  folder_id = "b1g6g0odfjqvpuhrneau"
  zone      = "ru-central1-a"
}



resource "yandex_compute_instance" "flu-vm-1" {
  name = "gpu-terraform"

  resources {
    cores  = 4
    memory = 8
    core_fraction = 100
    # gpus = 1
  }

  boot_disk {
    initialize_params {
      # image_id = "fd8dvtfpeskabitc3qlk"
      image_id = "fd8vpdk1hs9b1kuiiu13"
      size = 50
    }
  }

  network_interface {
    subnet_id = yandex_vpc_subnet.subnet-1.id
    nat       = true
  }

  metadata = {
    user-data = <<-EOF
    #cloud-config
    users:
      - name: flu
        group: sudo
        shell: /bin/bash
        sudo: ['ALL=(ALL) NOPASSWD:ALL']
        ssh-authorized-keys: ${file("/home/sergey/.ssh/id_rsa.pub")}

    EOF

    serial-port-enable = 1

  }
    # serial-port-enable = 1

  scheduling_policy {
      preemptible = true
    }

}

resource "yandex_vpc_network" "network-1" {
  name = "flu-network"
}

resource "yandex_vpc_subnet" "subnet-1" {
  name           = "flu-subnet"
  zone           = "ru-central1-a"
  network_id     = yandex_vpc_network.network-1.id
  v4_cidr_blocks = ["192.168.10.0/24"]
}

//output "internal_ip_address_vm_1" {
//  value = "yandex_compute_instance.vm-1.network_interface.0.ip_address"
//}

output "external_ip_address_vm_1" {
  value = yandex_compute_instance.flu-vm-1.network_interface.0.nat_ip_address
}
