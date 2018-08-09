procedure Test is
   procedure P (X : out Integer) with Import;

   X : Integer;
begin
   if False then
      P (X);
   end if;
end A;
