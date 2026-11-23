# Postmortem: caída del proveedor de pagos (14/10/2026)

> Documentación ficticia creada para la Pre-entrega 6.

## Resumen del incidente

El martes 14/10/2026, entre las 19:40 y las 22:55 (3 horas y 15 minutos), el proveedor externo de pagos respondió con timeouts. La tasa de errores 5xx del servicio de pagos llegó a un pico del **38%**. Se declaró **SEV1** a las 19:52 y se nombró comandante de incidente.

A las 20:05 se activó el feature flag `pagos_modo_cola`: desde ese momento los pagos se guardaron para procesarlos cuando el proveedor se recuperara, y los clientes vieron el mensaje "Tu pago se está procesando".

## Impacto en clientes

- **1.240 pedidos** quedaron en estado "procesando" durante el incidente; todos se cobraron antes de las 02:00 del 15/10.
- **312 clientes** fueron cobrados dos veces: antes de que se activara `pagos_modo_cola`, el botón "Pagar" seguía habilitado y muchos clientes lo apretaron de nuevo. Los reintegros se completaron el 18/10/2026.
- Los emails de confirmación de compra llegaron con hasta **6 horas** de demora, porque el worker de notificaciones procesó la cola acumulada.
- Los contactos a soporte se cuadruplicaron durante la semana siguiente.

## Causa raíz

El origen fue la caída del proveedor de pagos, pero el impacto se agravó por dos problemas internos:

1. El servicio de pagos no usaba **claves de idempotencia**: cada clic en "Pagar" generaba un cobro nuevo.
2. La alerta de errores de pagos se disparaba recién al 2% sostenido durante 5 minutos, así que `pagos_modo_cola` se activó 25 minutos después del primer error.

## Acciones

| Acción | Responsable | Fecha límite |
|---|---|---|
| Agregar claves de idempotencia a todos los cobros | Equipo Pagos | 31/10/2026 |
| Deshabilitar el botón "Pagar" mientras hay un cobro en curso | Equipo Checkout | 24/10/2026 |
| Bajar el umbral de alerta de pagos al 1% durante 2 minutos | Equipo Plataforma | 21/10/2026 |
| Enviar un email de disculpas con un cupón del 10% a los clientes con cobro doble | Atención al cliente | 20/10/2026 |
