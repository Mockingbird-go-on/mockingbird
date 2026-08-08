# Факт-база резюме (единый источник правды)

Источник: `docs/resume_gleb_devops.md`. Любое число/факт в ответах KB обязано
совпадать с этим документом. Метрики неприкосновенны.

## Сущности (работодатели)

| Компания | Период | Роль | Локация |
|---|---|---|---|
| NAUMEN | Март 2025 — наст. время (1 год 6 мес) | Инженер группы управления приложениями | Екатеринбург / удалённо |
| ООО Марс | Май 2023 — Февраль 2025 (1 год 10 мес) | Руководитель тех. отдела / Ведущий системный администратор | СПб |
| ТЕЛЕСТОР | Сентябрь 2022 — Апрель 2023 (8 мес) | Инженер технической поддержки (SIP) | СПб |
| Вегет | Февраль 2022 — Май 2022 (4 мес) | Зам. руководителя технической поддержки | СПб |
| Вегет | Ноябрь 2019 — Февраль 2022 (2 года 4 мес) | Ведущий инженер 2 линии технической поддержки | СПб |

Общий опыт: **6 лет 7 месяцев**. Должность: DevOps-инженер, удалённо, 165 000 ₽ на руки.
Образование: БГТУ «ВОЕНМЕХ» им. Д.Ф. Устинова (спец. «Стрелково-пушечное, артиллерийское
и ракетное оружие»). Английский B2. Сертификаты: VK Cloud Native Advanced, Яндекс Практикум.

## Технологии (из резюме)

**IaC/CI/CD:** GitLab CI, GitHub Actions, Ansible (роли, Jinja2, динамические инвентори),
Terraform, Vault, Workspaces, S3-state + блокировки.

**K8s/контейнеры:** Docker, Docker Compose, Helm-чарты, GitOps, ArgoCD (App-of-Apps, авто-синк,
webhook), Keycloak (SSO), RollingUpdate, Blue/Green.

**БД/кэши:** MariaDB, Patroni-кластер PostgreSQL, Redis, MS SQL, 1С платформа.

**Наблюдаемость:** Prometheus, VictoriaMetrics (VM Cluster, push-протокол, windows_exporter),
Grafana, Loki, Alertmanager, SLI/SLO, кастомные Python-экспортеры (метрики 1С).

**Windows/виртуализация:** Windows Server 2012r/2019/2022/2025, Hyper-V + S2D, AD (леса, GPO),
Exchange, WSUS, IIS, MS Exchange, KSMG (Kaspersky), Veeam, Кибер Бэкап, Cobian Backup, VMware, Proxmox.

**Сеть/DNS/почта:** Bind9, PowerDNS, DNS AD, Postfix/Dovecot, SPF/DKIM/DMARC, RouterOS, ZX-UI,
OpenVPN/WireGuard, SIP (ТЕЛЕСТОР), Wireshark/tcpdump.

**Приложения/веб:** Bitrix24, Nextcloud, Confluence, Jira, Netbox, RDM, 1С (в т.ч. Linux, HASP),
LAMP/LEMP, Wordpress/Joomla/Bitrix/MODx/PrestaShop, Nginx+Apache+PHP.

**Языки:** Python, PHP, Bash, PowerShell (скрипты, FastAPI-эндпоинты).

**Облака:** Cloud.ru, VK Cloud, Yandex Cloud.

## Метрики (неприкосновенны)

| Метрика | Значение | Где |
|---|---|---|
| Создание типовой инфраструктуры | 2 часа → ~15 минут, ручные ошибки исключены | NAUMEN, Terraform |
| Подготовка нового сервиса к выкатке | −50-60% | NAUMEN, Helm/K8s |
| MTTR (корреляция метрик и логов) | ~ −40% | NAUMEN, Loki/Grafana |
| Время эскалации (Runbooks) | ~ −30% | NAUMEN |
| Хосты через push-протокол | 50+ | NAUMEN, VictoriaMetrics Agent |
| Retention метрик | 1 год | Марс, VictoriaMetrics |
| Загрузка дашбордов Grafana | −50% | Марс, агрегация экспортеров |
| Нагрузка на мониторинг | ×3 ниже | Марс |
| Обращения техподдержки | 10 000+ | Вегет |
| Проходимость почты | ~100% (исключён спам) | Вегет |
| Резервное копирование | SLA RPO 15 мин (Veeam, критические системы) | Марс |
| Репликация MariaDB | задержка < 60 с (алерт Seconds_Behind_Master) | Марс |

## Проекты (ключевые)

1. **Гибридная DNS-система** (NAUMEN): Bind9 + PowerDNS + DNS AD, IaC через GitLab CI,
   тест зон перед деплоем, роллбэк.
2. **Terraform-модули Cloud.ru/VK Cloud** (NAUMEN): VPC, VM, K8s, БД; S3-state + блокировки;
   plan на MR / apply после ревью; Workspaces dev/stage/prod; Vault.
3. **Библиотека Helm-чартов + GitOps** (NAUMEN): унификация dev-prod, RollingUpdate/Blue-Green,
   ArgoCD App-of-Apps, авто-синк, webhook, SSO/RBAC на namespace.
4. **Стек наблюдаемости** (NAUMEN): Prometheus + VictoriaMetrics (кластер), windows_exporter,
   push для NAT-сегментов (50+ хостов), Loki + Grafana (MTTR −40%), алертинг по логам.
5. **Миграция мониторинга Zabbix → Prometheus+VictoriaMetrics** (Марс): retention 1 год,
   SLI/SLO + Alertmanager, кастомные Python-экспортеры (1С), дашборды −50%, нагрузка ×3 ниже.
6. **Ansible-плейбуки платформы 1С** (Марс): open source `github.com/glebgv/1c-ansible-platform`;
   под ключ: Nextcloud, ZX-UI, LAMP/LEMP, Bind9/Pdns/VPN, 1С-сервер, Postfix+Dovecot, Grafana.
7. **API-сервер SaaS 1С** (Марс): FastAPI принимал JSON от платформы → создание клиента, базы 1С,
   квот, папок → письмо с RDP-доступом к рабочему месту 1С.
8. **Почта и гибрид** (NAUMEN): MS Exchange + Postfix/Dovecot + KSMG (настройка/обновление/резерв).
9. **Runbooks + self-healing** (NAUMEN): эскалация −30%, авто-восстановление через Ansible/K8s.
10. **Курс сисадмина** (Яндекс Практикум): участие в разработке; сертификат VK Cloud Native Advanced.

## Матрица покрытия «пункт резюме → блок Q&A»

| Пункт резюме | Блоки (Q) |
|---|---|
| DNS Bind9+PowerDNS+DNS AD, IaC, GitLab CI | 1 |
| HA Windows+Linux | 2 |
| Terraform, S3-state, Vault, Workspaces, 2ч→15мин | 3, 4, 28 |
| Keycloak SSO | 9, 43 |
| Helm-библиотека, GitOps, Rolling/Blue-Green, −50-60% | 10, 11, 12, 44 |
| ArgoCD App-of-Apps, RBAC | 11, 61 |
| Patroni PostgreSQL, MariaDB, Redis | 18, 29, 30, 53 |
| VMware/Proxmox | 19, 36, 62 |
| Prometheus+VM, windows_exporter, push, 50+ хостов | 5, 46 |
| Loki+Grafana, MTTR −40%, алертинг по логам | 6, 7, 8 |
| Runbooks, self-healing, эскалация −30% | 13, 14 |
| Confluence/Jira/Nextcloud/AD/WSUS/Netbox/RDM/Bitrix24 | 24, 31, 34, 59, 63 |
| Архитектура, миграции | 17, 18 |
| Курс Практикума, сертификат VK | 15, 16, 49 |
| 1С облако, S2D, Hyper-V, Veeam/Кибер Бэкап, Cobian | 37, 38, 40, 52, 57, 64 |
| Мониторинг Zabbix→VM, SLI/SLO, экспортеры 1С, дашборды −50%, ×3 | 26, 27, 35 |
| Ansible под ключ, 1c-ansible-platform, FastAPI 1С SaaS | 21, 32, 38 |
| ZX-UI, VPN | 33 |
| SIP (ТЕЛЕСТОР) | 39 |
| Веб/CMS/Nginx+Apache+PHP, почта 100%, 10 000 обращений | 39, 45, 51 |
| Post-Mortems Blameless | 41 |
| Почта Postfix/Dovecot/Exchange/KSMG | 22, 23, 56 |
| Безопасность облака, security groups | 60 |
| DevOps-скрипты Python/Bash/PowerShell/jq | 21, 55 |
| Карьера/развитие/удалёнка | 50, 65 |

Обратное: каждый блок Q&A привязан к ≥1 пункту резюме (сирот нет).
