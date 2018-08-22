procedure Test is
   X : Integer;
   Y : Integer := 1;
   Z : Boolean;

   procedure A is
   begin
      X := 12;
   end A;

   procedure B(X : Boolean) is
      procedure C is
      begin
         X := 14;
      end C;

      P : access procedure := null;
   begin
      if X then
         P := A'Access;
      else
         P := C'Access;
      end if;
      P.all;
   end B;
begin
   B(Z);
end Test;
